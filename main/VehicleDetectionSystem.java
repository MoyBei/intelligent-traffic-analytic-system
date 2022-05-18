package ai.certifai;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.common.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

public class VehicleDetectionSystem {

    public static final Logger log = LoggerFactory.getLogger(VehicleDetectionSystem.class);

    private static final String modelFilename = new File(".").getAbsolutePath() + "/generated-models/Model4_Final.zip"; //My_YOLO_Model_v2_25

    private static final String testImagePath = "E:\\Codes\\test\\test0.jpg";
    private static final String outputImagePath = "E:\\Codes\\test\\test0-output.jpg";
    private static final String testVideoPath = "E:\\Codes\\Java\\cdle-traininglabs-main\\my-first-dl4j-project\\src\\main\\not_resources\\test\\testvid1-10m-1.mp4";
    private static final String outputVideoPath = "E:\\Codes\\Java\\cdle-traininglabs-main\\my-first-dl4j-project\\src\\main\\not_resources\\test\\testvid1-10m-1-counting.mp4";
    private static final boolean isToRenderVideo = false;
    private static final boolean isToRenderBBox = true;

    public static void main(String[] args) throws Exception {
        // Setup Model
        File imageDir = new File(new ClassPathResource("/dataset").getFile().getPath(), "trainvidimg"); //train
        ModelConfig config = new ModelConfig();
        config.defaultConfig();

        // Load Model
        ModelLoader loader = new ModelLoader();
        ComputationGraph yoloTransfer = loader.loadModel(modelFilename, imageDir, config);

        // Test on Model
        ModelTester tester = new ModelTester();
//        tester.testOneImage(testImagePath, outputImagePath, yoloTransfer, config, isToRenderBBox);
        tester.testVideo(testVideoPath, outputVideoPath, yoloTransfer, config, isToRenderVideo, isToRenderBBox);

        // ----- End of Program ----- //
    }
}
