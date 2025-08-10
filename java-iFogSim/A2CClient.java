package org.fog.utils;

import java.io.*;
import java.net.*;
import org.json.simple.*;
import org.json.simple.parser.*;

/**
 * A2CClient
 * ---------
 * Purpose:
 *   Acts as the Java-side client for sending a task state vector to the A2C inference
 *   server (Python) and receiving the predicted cloud index for execution.
 *
 * Protocol:
 *   Java Client -> A2C Server (Python):
 *       JSON string containing:
 *           - state: direct JSON array of floats (not stringified)
 *             Example: {"state": [0.72, 0.33, 0.15, 1.0, 0.28]}
 *
 *   A2C Server -> Java Client:
 *       JSON string containing:
 *           - cloud: integer cloud index (0 = AWS, 1 = Azure, 2 = GCP, etc.)
 *             Example: {"cloud": 1}
 *
 * Notes:
 *   - Unlike PPOClient, there is no reward/next_state here — this is a pure
 *     inference call without training feedback.
 *   - Uses direct JSON arrays for the state vector to match the A2C server format.
 *   - The TCP connection is opened, used for a single request/response, and closed.
 *   - Defaults to returning 0 if an error occurs.
 */
public class A2CClient {

    /**
     * Sends a state vector to the A2C inference server and returns the chosen cloud index.
     * @param stateVec double[] of current state features
     * @return predicted cloud index
     */
    public static int getA2CDecision(double[] stateVec) {
        int cloud = 0; // Default to AWS if failure

        try {
            Socket socket = new Socket("localhost", 9999); // Match A2C server host/port
            OutputStream output = socket.getOutputStream();
            InputStream input = socket.getInputStream();

            // Build JSON request payload
            JSONObject request = new JSONObject();
            JSONArray state = new JSONArray();
            for (double v : stateVec) state.add(v);
            request.put("state", state);

            // Send request to server
            PrintWriter writer = new PrintWriter(output, true);
            writer.println(request.toJSONString());

            // Read response from server
            BufferedReader reader = new BufferedReader(new InputStreamReader(input));
            String response = reader.readLine();

            // Parse JSON response and extract cloud index
            JSONParser parser = new JSONParser();
            JSONObject result = (JSONObject) parser.parse(response);
            cloud = ((Long) result.get("cloud")).intValue();

            socket.close();
        } catch (Exception e) {
            System.out.println("⚠️ Error in A2CClient: " + e.getMessage());
        }

        return cloud;
    }
}
