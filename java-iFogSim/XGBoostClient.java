package org.fog.utils;

import java.io.*;
import java.net.*;
import java.util.*;

public class XGBoostClient {
    private static final String SERVER_HOST = "localhost";
    private static final int SERVER_PORT = 65432;

    public static int getPrediction(List<Double> features) {
        try (Socket socket = new Socket(SERVER_HOST, SERVER_PORT)) {
            // Convert feature list to JSON
            StringBuilder json = new StringBuilder("[");
            for (int i = 0; i < features.size(); i++) {
                json.append(features.get(i));
                if (i != features.size() - 1) json.append(",");
            }
            json.append("]");

            // Send features
            OutputStream output = socket.getOutputStream();
            output.write(json.toString().getBytes());
            output.flush();

            // Receive prediction
            BufferedReader reader = new BufferedReader(new InputStreamReader(socket.getInputStream()));
            String response = reader.readLine();
            return Integer.parseInt(response.trim());

        } catch (IOException e) {
            System.err.println("Error communicating with XGBoost server: " + e.getMessage());
            return 0; // default to cloud 0 (AWS) if error
        }
    }
}
