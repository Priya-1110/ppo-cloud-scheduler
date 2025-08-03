package org.fog.scheduler;

import org.fog.entities.FogDevice;
import java.util.List;

import java.util.Random;

public class FCFScheduler {
    public static int getLeastLoadedCloudIndex(List<FogDevice> cloudProviders) {
        Random rand = new Random();
        int selectedIndex = 0;
        double minUtilization = Double.MAX_VALUE;

        for (int i = 0; i < cloudProviders.size(); i++) {
            double fakeUtil = rand.nextDouble(); // 0.0 to 1.0
            if (fakeUtil < minUtilization) {
                minUtilization = fakeUtil;
                selectedIndex = i;
            }
        }
        return selectedIndex;
    }
}
