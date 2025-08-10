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
import org.fog.scheduler.RoundRobinScheduler;
import org.fog.scheduler.FCFScheduler;
import org.fog.scheduler.StreamOperatorScheduler;
import org.fog.utils.*;
import org.fog.placement.*;
import org.fog.utils.distribution.DeterministicDistribution;

/**
 * MultiCloudSchedulingSim
 * -----------------------
 * Purpose:
 *   Simulates a multi-cloud scheduling environment using the iFogSim framework,
 *   allowing evaluation of different scheduling algorithms for IoT/cloud workloads.
 *
 * Supported Scheduling Models:
 *   - PPO (Proximal Policy Optimization) via PPOClient (Python socket server, port 5055)
 *   - A2C (Advantage Actor-Critic) via A2CClient (Python socket server, port 9999)
 *   - DQN (Deep Q-Network) via DQNClient (Python socket server, port 9999)
 *   - Round Robin (static baseline) via RoundRobinScheduler
 *   - FCFS (First-Come-First-Serve) via FCFScheduler
 *
 * Switching Models:
 *   To switch between models, comment out the currently active scheduler line
 *   and uncomment the desired one in the `processTupleArrival` method.
 *   Also adjust the CSV output filename at the top of main().
 *
 * Output:
 *   Logs simulation results (TaskID, selected cloud, timings, cost, SLA status)
 *   into a CSV file for later analysis.
 */
public class MultiCloudSchedulingSim {

    static List<FogDevice> cloudProviders = new ArrayList<>();
    static List<Sensor> sensors = new ArrayList<>();
    static List<Actuator> actuators = new ArrayList<>();
    static Map<String, Integer> nameToIdMap = new HashMap<>();
    static Map<Integer, FogDevice> idToDeviceMap = new HashMap<>();

    static FileWriter csvWriter;

    public static void main(String[] args) {
        Log.printLine("=== Starting Multi-Cloud Scheduling Simulation ===");
        try {
            // Select CSV log file based on model being run
            //csvWriter = new FileWriter("round_robin_log.csv");
            //csvWriter = new FileWriter("fcfs_log.csv");
            //csvWriter = new FileWriter("A2C_log.csv");
            csvWriter = new FileWriter("ppo_log.csv"); // Active: PPO
            //csvWriter = new FileWriter("dqn_log.csv");

            csvWriter.append("TaskID,SelectedCloud,StartTime,EndTime,ExecutionTime,CPUCost,SLADuration,SLAMet\n");

            // Initialize CloudSim with one user and no trace events
            CloudSim.init(1, Calendar.getInstance(), false);
            FogBroker broker = new FogBroker("broker");

            Map<Integer, String> idMap = new HashMap<>();

            // Create cloud provider FogDevices
            FogDevice aws = createCloudProvider("AWS_Cloud", 10000, 16384, 10000, 10000, 0.10, 100.0, idMap);
            FogDevice azure = createCloudProvider("Azure_Cloud", 7000, 8192, 8000, 8000, 0.07, 150.0, idMap);
            FogDevice gcp = createCloudProvider("GCP_Cloud", 5000, 8192, 6000, 6000, 0.05, 200.0, idMap);

            // Register cloud providers
            cloudProviders.addAll(Arrays.asList(aws, azure, gcp));
            for (FogDevice cloud : cloudProviders) {
                cloud.setParentId(-1);
                CloudSim.addEntity(cloud);
                idToDeviceMap.put(cloud.getId(), cloud);
            }

            // Create application graph
            Application app = createApp("multi_cloud_app", broker.getId());

            // Create sensors and actuators
            for (int i = 0; i < 10; i++) {
                createSensorAndActuator("sensor" + i, "actuator" + i, broker.getId(), aws.getId(), app, FogUtils.generateRandomDouble(3, 7));
            }

            nameToIdMap.put("AWS_Cloud", aws.getId());
            nameToIdMap.put("Azure_Cloud", azure.getId());
            nameToIdMap.put("GCP_Cloud", gcp.getId());

            for (Sensor sensor : sensors) {
                sensor.setApp(app);
            }

            // Configure controller for clustering
            LocationHandler locator = new StaticLocationHandler(idMap);
            ClusteringController controller = new ClusteringController("controller", cloudProviders, sensors, actuators, locator);

            ModuleMapping moduleMapping = ModuleMapping.createModuleMapping();
            controller.submitApplication(app, 0, new ModulePlacementMapping(cloudProviders, app, moduleMapping));

            Log.printLine("Created cloud providers + IoT devices");

            // Run simulation
            CloudSim.startSimulation();
            CloudSim.stopSimulation();

            // Save and close CSV
            csvWriter.flush(); 
            csvWriter.close();

            Log.printLine("=== Simulation Finished ===");

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

        FogDeviceCharacteristics characteristics = new FogDeviceCharacteristics(
            "x86", "Linux", "Xen", host, 10.0, costPerMips, 0.05, 0.001, 0.0);

        FogDevice device = null;
        try {
            device = new FogDevice(name, characteristics, new AppModuleAllocationPolicy(hostList),
                new LinkedList<>(), 10, upBw, downBw, 0, latency) {

                private int tupleCounter = 1;

                @Override
                public void processTupleArrival(org.cloudbus.cloudsim.core.SimEvent ev) {
                    Tuple tuple = (Tuple) ev.getData();

                    // Generate random CPU and memory demands for the task
                    long cpuDemand = (long) FogUtils.generateRandomDouble(7000, 10000);
                    long memDemand = (long) FogUtils.generateRandomDouble(128, 1024);
                    double slaDeadline = cpuDemand / 8000.0; 

                    double startTime = CloudSim.clock();

                    // State vector (length 5)
                    double[] state = {
                        cpuDemand / 10000.0,
                        memDemand / 1024.0,
                        startTime / 1000.0,
                        slaDeadline / 10.0,
                        0.0
                    };

                    // =============================
                    // 🔹 SELECT SCHEDULER HERE
                    // =============================

                    // PPO (active)
                    int taskId = tupleCounter;
                    int selectedCloud = PPOClient.getPPOAction(
                        taskId, state, 0.0, false, state, 0.0,
                        "PENDING", slaDeadline, 0.0
                    );

                    // Round Robin
                    //int selectedCloud = RoundRobinScheduler.getNextCloudIndex(cloudProviders);

                    // FCFS
                    //int selectedCloud = FCFScheduler.getLeastLoadedCloudIndex(cloudProviders);

                    // A2C
                    //int selectedCloud = A2CClient.getA2CDecision(state);

                    // DQN
                    //int selectedCloud = DQNClient.getDQNAction(state);

                    // =============================

                    FogDevice selectedDevice = cloudProviders.get(selectedCloud);
                    double execTime = cpuDemand / selectedDevice.getHost().getTotalMips();
                    double endTime = startTime + execTime;
                    double actualCost = cpuDemand * selectedDevice.getRatePerMips() / 10000.0;
                    boolean slaMet = execTime <= slaDeadline;

                    // Reward shaping (for RL-based schedulers)
                    double reward = 1.0;
                    if (!slaMet) reward -= 1.5;
                    reward -= actualCost / 10.0;

                    // Next state (reduced demands)
                    double[] nextState = {
                        (cpuDemand * 0.9) / 10000.0,
                        (memDemand * 0.9) / 1024.0,
                        (startTime + execTime) / 1000.0,
                        slaDeadline / 10.0,
                        actualCost / 10.0
                    };
                    
                    // ✅ Debug console log
                    //System.out.println("🔁 RR → TaskID: " + taskId + ", Cloud: " + selectedCloud + ", SLA: " + (slaMet ? "YES" : "NO") + ", Reward: " + reward);
                    //System.out.println("🔁 FCFS → TaskID: " + taskId + ", Cloud: " + selectedCloud + ", SLA: " + (slaMet ? "YES" : "NO") + ", Reward: " + reward);
                    //System.out.println("🔁 A2C → TaskID: " + taskId + ", Cloud: " + selectedCloud + ", SLA: " + (slaMet ? "YES" : "NO") + ", Reward: " + reward);
                    //System.out.println("🔁 DQN → TaskID: " + taskId + ", Cloud: " + selectedCloud + ", SLA: " + (slaMet ? "YES" : "NO") + ", Reward: " + reward);
                    System.out.println("🔁 PPO → TaskID: " + taskId + ", Cloud: " + selectedCloud + ", SLA: " + (slaMet ? "YES" : "NO") + ", Reward: " + reward);

                    tuple.setCloudletLength(cpuDemand);
                    tuple.setUserId((int) memDemand);
                    tuple.setDestinationDeviceId(selectedDevice.getId());

                    // Write to CSV
                    try {
                        csvWriter.append(tupleCounter++ + "," + selectedCloud + "," +
                            startTime + "," + endTime + "," + execTime + "," +
                            actualCost + "," + slaDeadline + "," +
                            (slaMet ? "YES" : "NO") + "\n");
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
