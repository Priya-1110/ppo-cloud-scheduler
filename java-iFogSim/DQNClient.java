package org.fog.utils;

import java.io.*;
import java.net.*;
import org.json.simple.*;
import org.json.simple.parser.*;

/**
 * DQNClient
 * ---------
 * Purpose:
 *   Java-side client for interacting with the DQN inference server (Python).
 *   Sends the current state vector over TCP and receives the predicted action
 *   (cloud index) from the trained DQN model.
 *
 * Protocol:
 *   Java Client -> DQN Server (Python):
 *       JSON string containing:
 *           - state: direct JSON array of floats
 *             Example: {"state": [0.82, 0.41, 0.22, 1.0, 0.37]}
 *
 *   DQN Server -> Java Client:
 *       JSON string containing:
 *           - action: integer cloud index
 *             Example: {"action": 2}
 *
 * Notes:
 *   - Unlike PPOClient, this client sends only the state vector (no reward or next state).
 *   - Uses direct JSON arrays for the state vector to match DQN server format.
 *   - Defaults to returning 0 if any error occurs during communication.
 *   - The TCP connection is opened per request and closed after the response.
 */
public class DQNClient {

    /**
     * Sends a state vector to the DQN inference server and returns the chosen cloud index.
     * @param stateVec double[] representing the current task state features
     * @return predicted cloud index (0, 1, 2, etc.)
     */
    public static int getDQNAction(double[] stateVec) {
        int action = 0;  // Default to cloud index 0 if error

        try {
            Socket socket = new Socket("localhost", 9999); // Match DQN server host/port
            OutputStream output = socket.getOutputStream();
            InputStream input = socket.getInputStream();

            // Build JSON request: { "state": [...] }
            JSONObject request = new JSONObject();
            JSONArray state = new JSONArray();
            for (double v : stateVec) state.add(v);
            request.put("state", state);

            // Send request to Python DQN server
            PrintWriter writer = new PrintWriter(output, true);
            writer.println(request.toJSONString());

            // Read response from server
            BufferedReader reader = new BufferedReader(new InputStreamReader(input));
            String response = reader.readLine();

            // Parse JSON: { "action": 0 }
            JSONParser parser = new JSONParser();
            JSONObject result = (JSONObject) parser.parse(response);
            action = ((Long) result.get("action")).intValue();

            socket.close();
        } catch (Exception e) {
            System.out.println("⚠️ Error in DQNClient: " + e.getMessage());
        }

        return action;
    }
}
