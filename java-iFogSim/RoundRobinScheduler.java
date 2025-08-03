package org.fog.scheduler;

import org.fog.entities.FogDevice;
import java.util.List;

public class RoundRobinScheduler {
    private static int rrIndex = 0;

    public static int getNextCloudIndex(List<FogDevice> cloudProviders) {
        if (cloudProviders.isEmpty()) return 0;
        int index = rrIndex;
        rrIndex = (rrIndex + 1) % cloudProviders.size();
        return index;
    }
}
