package org.fog.utils;

import java.io.*;
import java.net.*;
import org.json.simple.JSONObject;

/**
 * PPOClient
 * ---------
 * Purpose:
 *   Acts as a Java-side client for sending scheduling requests to the PPO inference
 *   server over TCP. Transmits task metadata and state vectors, then receives the
 *   predicted cloud index for execution.
 *
 * Protocol:
 *   Java Client -> PPO Server (Python):
 *       JSON string containing:
 *           - task_id         : unique ID of the task
 *           - state           : stringified JSON list of floats (current state vector)
 *           - reward          : last step reward value
 *           - done            : episode completion flag
 *           - next_state      : stringified JSON list of floats (next state vector)
 *           - cost            : CPU execution cost for the task
 *           - sla_met         : "YES"/"NO" SLA compliance indicator
 *           - sla_deadline    : deadline in ms
 *           - execution_time  : actual execution time in ms
 *
 *   PPO Server -> Java Client:
 *       String containing an integer ("0", "1", "2", ...) for the selected cloud index.
 *
 * Notes:
 *   - state and next_state are double arrays converted to a *stringified JSON array*
 *     to match the PPO server’s expected parsing format.
 *   - The TCP connection is opened, used for one request/response, and then closed.
 *   - Defaults to returning 0 (AWS) if there is an error communicating with the server.
 */
public class PPOClient {

    private static final String SERVER_HOST = "localhost"; // PPO server host
    private static final int SERVER_PORT = 5055;           // PPO server port

    /**
     * Sends the task state and metadata to the PPO inference server and returns the predicted action.
     * @return integer cloud index chosen by PPO
     */
    public static int getPPOAction(int taskId, double[] state, double reward, boolean done,
                                   double[] nextState, double cost, String slaMet,
                                   double slaDeadline, double execTime) {

        try (Socket socket = new Socket(SERVER_HOST, SERVER_PORT)) {
            // Output writer and input reader for the TCP connection
            PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
            BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));

            // Build JSON payload for PPO server
            JSONObject payload = new JSONObject();
            payload.put("task_id", taskId);
            payload.put("state", stateToString(state));       // Stringified list format
            payload.put("reward", reward);
            payload.put("done", done);
            payload.put("next_state", stateToString(nextState));
            payload.put("cost", cost);
            payload.put("sla_met", slaMet);
            payload.put("sla_deadline", slaDeadline);
            payload.put("execution_time", execTime);

            // Send JSON to server
            out.println(payload.toJSONString());

            // Receive cloud index (string) from server
            String response = in.readLine();
            return Integer.parseInt(response);

        } catch (IOException e) {
            e.printStackTrace();
            return 0; // Default to AWS (0) if error occurs
        }
    }

    /**
     * Converts a double[] to a stringified JSON array, e.g., "[0.50, 0.30, 0.20, 0.10, 0.00]".
     */
    private static String stateToString(double[] arr) {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < arr.length; i++) {
            sb.append(String.format("%.5f", arr[i]));
            if (i != arr.length - 1) sb.append(", ");
        }
        sb.append("]");
        return sb.toString();
    }
}
