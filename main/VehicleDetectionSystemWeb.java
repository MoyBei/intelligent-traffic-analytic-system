package ai.certifai;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.common.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;

public class VehicleDetectionSystemWeb {

    public static final Logger log = LoggerFactory.getLogger(VehicleDetectionSystemWeb.class);

    private static final String modelFilename = new File(".").getAbsolutePath() + "/generated-models/Model4_Final.zip"; //My_YOLO_Model_v2_25

    private static final String testImageFolderPath = "E:\\Codes\\Frontend\\inputImages";
    private static final String outputImageFolderPath = "E:\\Codes\\Frontend\\outputImages";
    private static boolean isToRenderBBox;

    private static ModelTester tester;
    private static ModelConfig config;
    private static ComputationGraph yoloTransfer;

    public static void main(String[] args) throws Exception {
        // Setup Model
        File imageDir = new File(new ClassPathResource("/dataset").getFile().getPath(), "trainvidimg"); //train
        config = new ModelConfig();
        config.defaultConfig();

        // Load Model
        ModelLoader loader = new ModelLoader();
        yoloTransfer = loader.loadModel(modelFilename, imageDir, config);

        // Load Tester
        tester = new ModelTester();

        // Load Server
        HttpServer server = HttpServer.create(new InetSocketAddress(8001), 0);
        server.createContext("/poke", new PokeHandler());
        server.setExecutor(null);
        server.start();

        // ----- End of Program ----- //
    }

    private static class PokeHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            String response = "";

            if (!exchange.getRequestMethod().startsWith("GET")) {
                exchange.sendResponseHeaders(400, 0);
                OutputStream outputStream = exchange.getResponseBody();
                outputStream.write(response.getBytes(StandardCharsets.UTF_8));
                outputStream.close();
            }

            Map<String, String> getArgs = new HashMap<>();
            for (String getArg : exchange.getRequestURI().getQuery().split("&")) {
                String[] pair = getArg.split("=");
                if (pair.length > 1) getArgs.put(pair[0], pair[1]);
                else getArgs.put(pair[0], "");
            }

            String inputFile = getArgs.getOrDefault("inputFile", null);
            String drawBoundingBox = getArgs.getOrDefault("drawBoundingBox", "false");
            isToRenderBBox = Objects.equals(drawBoundingBox, "true");
            UUID uuid = UUID.randomUUID();

            String testImagePath = testImageFolderPath + "\\" + inputFile;
            String outputImagePath = outputImageFolderPath + "\\" + uuid + inputFile;

            tester.testOneImage(testImagePath, outputImagePath, yoloTransfer, config, isToRenderBBox);

            // Customize response to return output path instead.
            response = "/outputImages/" + uuid + inputFile;

            exchange.sendResponseHeaders(200, response.getBytes().length);
            OutputStream outputStream = exchange.getResponseBody();
            outputStream.write(response.getBytes(StandardCharsets.UTF_8));
            outputStream.close();
        }
    }
}
