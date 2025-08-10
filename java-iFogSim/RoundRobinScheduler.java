package org.fog.scheduler;

import org.fog.entities.FogDevice;
import java.util.List;

/**
 * RoundRobinScheduler
 * -------------------
 * Purpose:
 *   Implements a simple Round Robin (RR) baseline scheduler that cycles
 *   through all available cloud providers in sequence, assigning the next
 *   task to the next provider in the list.
 *
 * Notes:
 *   - Stateless with respect to task characteristics (does not consider SLA, cost, etc.).
 *   - Uses a static counter (rrIndex) to persist the round-robin position across calls.
 *   - Wraps around to the first provider when the end of the list is reached.
 */
public class RoundRobinScheduler {
    // Keeps track of the current position in the cloudProviders list
    private static int rrIndex = 0;

    /**
     * Returns the index of the next cloud provider in round-robin order.
     * @param cloudProviders list of candidate FogDevice clouds
     * @return selected cloud index
     */
    public static int getNextCloudIndex(List<FogDevice> cloudProviders) {
        if (cloudProviders.isEmpty()) return 0; // Default to index 0 if no providers
        int index = rrIndex;                    // Select current position
        rrIndex = (rrIndex + 1) % cloudProviders.size(); // Move to next position
        return index;
    }
}
