package ai.certifai;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class ModelConfig {
    public Random rng;
    public int seed;

    // Yolo Config
    public int height;
    public int width;
    public int nChannels;
    public int gridHeight;
    public int gridWidth;

    public int batchSize;
    public int nBoxes;
    public int nClasses;
    public int nEpochs;
    public double[][] priorBoxes;
    public double learningRate;
    public double lambdaNoObj;
    public double lambdaCoord;
    public double detectionThreshold;
    public List<String> labels;

    public void defaultConfig(){
        this.width = 416;
        this.height = 416;
        this.nChannels = 3;
        this.gridWidth = 13;
        this.gridHeight = 13;
        // parameters for the Yolo2OutputLayer
        this.nBoxes = 5;
        this.lambdaNoObj = 0.5;
        this.lambdaCoord = 5.0;
        this.priorBoxes = new double[][]{{0.5727, 0.6774}, {1.8745, 2.0625}, {3.3384, 5.4743}, {7.8828, 3.5278}, {9.7705, 9.1683}};
        this.detectionThreshold = 0.3;
        // parameters for the training phase
        this.batchSize = 4;
        this.nEpochs = 50;
        this.learningRate = 1e-4;
        this.nClasses = 3;
        this.seed = 123;
        this.rng = new Random(seed);
        this.labels = Arrays.asList("car", "commercial", "motorbike");
    }
}
