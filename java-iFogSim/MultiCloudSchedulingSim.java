package org.fog.test;

import java.io.*;
import java.util.*;

import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.Pe;
import org.cloudbus.cloudsim.sdn.overbooking.PeProvisionerOverbooking;
import org.cloudbus.cloudsim.power.PowerHost;
import org.cloudbus.cloudsim.sdn.overbooking.BwProvisionerOverbooking;
import org.cloudbus.cloudsim.provisioners.RamProvisionerSimple;
import org.cloudbus.cloudsim.Host;
import org.cloudbus.cloudsim.Log;

import org.fog.application.Application;
import org.fog.application.AppEdge;
import org.fog.application.AppLoop;
import org.fog.application.AppModule;
import org.fog.application.selectivity.FractionalSelectivity;
import org.fog.entities.*;
import org.fog.policy.AppModuleAllocationPolicy;
//import org.fog.scheduler.RoundRobinScheduler;
import org.fog.scheduler.FCFScheduler;
import org.fog.scheduler.StreamOperatorScheduler;
import org.fog.utils.*;
import org.fog.placement.*;
import org.fog.utils.distribution.DeterministicDistribution;

public class MultiCloudSchedulingSim {

    static List<FogDevice> cloudProviders = new ArrayList<>();
    static List<Sensor> sensors = new ArrayList<>();
    static List<Actuator> actuators = new ArrayList<>();
    static Map<String, Integer> nameToIdMap = new HashMap<>();
    static Map<Integer, FogDevice> idToDeviceMap = new HashMap<>();

    static FileWriter csvWriter;

    public static void main(String[] args) {
        Log.printLine("=== Starting Multi-Cloud Scheduling Simulation with Round Robin ===");
        try {
            //csvWriter = new FileWriter("round_robin_log.csv");
            csvWriter = new FileWriter("fcfs_log.csv");
            csvWriter.append("TaskID,SelectedCloud,StartTime,EndTime,ExecutionTime,CPUCost,SLADuration,SLAMet\n");

            CloudSim.init(1, Calendar.getInstance(), false);
            FogBroker broker = new FogBroker("broker");

            Map<Integer, String> idMap = new HashMap<>();

            FogDevice aws = createCloudProvider("AWS_Cloud", 10000, 16384, 10000, 10000, 0.10, 100.0, idMap);
            FogDevice azure = createCloudProvider("Azure_Cloud", 7000, 8192, 8000, 8000, 0.07, 150.0, idMap);
            FogDevice gcp = createCloudProvider("GCP_Cloud", 5000, 8192, 6000, 6000, 0.05, 200.0, idMap);

            cloudProviders.addAll(Arrays.asList(aws, azure, gcp));
            for (FogDevice cloud : cloudProviders) {
                cloud.setParentId(-1);
                CloudSim.addEntity(cloud);
                idToDeviceMap.put(cloud.getId(), cloud);
            }

            Application app = createApp("multi_cloud_app", broker.getId());

            for (int i = 0; i < 10; i++) {
                createSensorAndActuator("sensor" + i, "actuator" + i, broker.getId(), aws.getId(), app, FogUtils.generateRandomDouble(3, 7));
            }

            nameToIdMap.put("AWS_Cloud", aws.getId());
            nameToIdMap.put("Azure_Cloud", azure.getId());
            nameToIdMap.put("GCP_Cloud", gcp.getId());

            for (Sensor sensor : sensors) {
                sensor.setApp(app);
            }

            LocationHandler locator = new StaticLocationHandler(idMap);
            ClusteringController controller = new ClusteringController("controller", cloudProviders, sensors, actuators, locator);

            ModuleMapping moduleMapping = ModuleMapping.createModuleMapping();
            controller.submitApplication(app, 0, new ModulePlacementMapping(cloudProviders, app, moduleMapping));

            Log.printLine("Created cloud providers + IoT devices");

            CloudSim.startSimulation();
            CloudSim.stopSimulation();

            csvWriter.flush(); csvWriter.close();

            Log.printLine("=== Round Robin Simulation Finished ===");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static FogDevice createCloudProvider(String name, int mips, long ram, long upBw, long downBw, double costPerMips, double latency, Map<Integer, String> idMap) {
        List<Pe> peList = new ArrayList<>();
        peList.add(new Pe(0, new PeProvisionerOverbooking(mips)));

        int hostId = FogUtils.generateEntityId();
        long storage = 1000000;

        PowerHost host = new PowerHost(hostId,
            new RamProvisionerSimple((int) ram),
            new BwProvisionerOverbooking((int) upBw),
            storage, peList,
            new StreamOperatorScheduler(peList),
            new FogLinearPowerModel(100, 50));

        List<Host> hostList = new ArrayList<>();
        hostList.add(host);

        FogDeviceCharacteristics characteristics = new FogDeviceCharacteristics("x86", "Linux", "Xen", host, 10.0, costPerMips, 0.05, 0.001, 0.0);

        FogDevice device = null;
        try {
            device = new FogDevice(name, characteristics, new AppModuleAllocationPolicy(hostList), new LinkedList<>(), 10, upBw, downBw, 0, latency) {
                private int tupleCounter = 1;

                @Override
                public void processTupleArrival(org.cloudbus.cloudsim.core.SimEvent ev) {
                    Tuple tuple = (Tuple) ev.getData();

                    long cpuDemand = (long) FogUtils.generateRandomDouble(7000, 10000);
                    long memDemand = (long) FogUtils.generateRandomDouble(128, 1024);
                    double slaDeadline = cpuDemand / 8000.0;
                    double startTime = CloudSim.clock();

                    // 🌐 ROUND ROBIN LOGIC
                    // int selectedCloud = RoundRobinScheduler.getNextCloudIndex(cloudProviders);
                    // 🌐 FCFS LOGIC
                    int selectedCloud = FCFScheduler.getLeastLoadedCloudIndex(cloudProviders);
                    FogDevice selectedDevice = cloudProviders.get(selectedCloud);

                    double selectedMips = selectedDevice.getHost().getTotalMips();
                    double execTime = cpuDemand / selectedMips;
                    double endTime = startTime + execTime;
                    double cost = cpuDemand * selectedDevice.getRatePerMips() / 10000;
                    boolean slaMet = execTime <= slaDeadline;

                    tuple.setCloudletLength(cpuDemand);
                    tuple.setUserId((int) memDemand);
                    tuple.setDestinationDeviceId(selectedDevice.getId());

                    try {
                        csvWriter.append(tupleCounter++ + "," + selectedCloud + "," +
                                startTime + "," + endTime + "," + execTime + "," +
                                cost + "," + slaDeadline + "," + (slaMet ? "YES" : "NO") + "\n");
                    } catch (IOException e) {
                        e.printStackTrace();
                    }

                    super.processTupleArrival(ev);
                }
            };
        } catch (Exception e) {
            e.printStackTrace();
        }

        device.setLevel(0);
        device.setName(name);
        idMap.put(device.getId(), name);
        return device;
    }

    private static Application createApp(String appId, int userId) {
        Application application = Application.createApplication(appId, userId);
        application.addAppModule("cloudModule", 10);
        application.addAppEdge("SENSOR", "cloudModule", 3000, 500, "SENSOR", Tuple.UP, AppEdge.SENSOR);
        application.addAppEdge("cloudModule", "Actuator", 1000, 100, "RESULT", Tuple.DOWN, AppEdge.ACTUATOR);
        application.addTupleMapping("cloudModule", "SENSOR", "RESULT", new FractionalSelectivity(1.0));
        application.addAppLoop(Arrays.asList("Sensor", "cloudModule", "Actuator"));
        return application;
    }

    private static void createSensorAndActuator(String sensorName, String actuatorName, int userId, int gatewayId, Application app, double interval) {
        Sensor sensor = new Sensor(sensorName, "SENSOR", userId, "multi_cloud_app", new DeterministicDistribution(interval));
        Actuator actuator = new Actuator(actuatorName, userId, "multi_cloud_app", actuatorName);
        sensor.setApp(app);
        sensor.setGatewayDeviceId(gatewayId);
        actuator.setGatewayDeviceId(gatewayId);
        sensor.setLatency(1.0);
        actuator.setLatency(1.0);
        sensors.add(sensor);
        actuators.add(actuator);
        CloudSim.addEntity(sensor);
        CloudSim.addEntity(actuator);
    }
}
