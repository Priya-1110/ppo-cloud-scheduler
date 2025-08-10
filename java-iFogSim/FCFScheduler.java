package org.fog.scheduler;

import org.fog.entities.FogDevice;
import java.util.List;
import java.util.Random;

/**
 * FCFScheduler (Simplified)
 * -------------------------
 * Purpose:
 *   Selects a target cloud index for the next task using a minimal heuristic.
 *   This implementation uses a placeholder "utilization" (random in [0,1)) to
 *   mimic "least-loaded" selection for a basic FCFS baseline in experiments.
 *
 * Notes:
 *   - This is NOT a true FCFS queueing system; it’s a lightweight, stateless
 *     chooser to keep the baseline simple and reproducible.
 *   - Replace the random 'fakeUtil' with real utilization metrics if available
 *     (e.g., host CPU utilization) to make "least-loaded" meaningful.
 */
public class FCFScheduler {

    /**
     * Returns the index of the cloud provider with the lowest (placeholder) utilization.
     * @param cloudProviders list of candidate FogDevice clouds
     * @return selected cloud index
     */
    public static int getLeastLoadedCloudIndex(List<FogDevice> cloudProviders) {
        Random rand = new Random();
        int selectedIndex = 0;
        double minUtilization = Double.MAX_VALUE;

        // Iterate and pick the smallest "utilization" (currently randomized)
        for (int i = 0; i < cloudProviders.size(); i++) {
            double fakeUtil = rand.nextDouble(); // TODO: replace with real metric
            if (fakeUtil < minUtilization) {
                minUtilization = fakeUtil;
                selectedIndex = i;
            }
        }
        return selectedIndex;
    }
}
