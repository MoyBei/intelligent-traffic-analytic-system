package ai.certifai;

import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.YOLO2;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;

import java.io.File;

public class ModelLoader {
    public ComputationGraph loadModel(String modelFilename, File imageDir, ModelConfig config) throws Exception {
        ComputationGraph model;

        if (new File(modelFilename).exists()) {
            model = loadTrainedModel(modelFilename);
        } else {
            VehicleDetectionSystem.log.info("File does not exist, training a model...");
            model = loadTrainedModel(
                    trainModel(
                            loadDataset(imageDir, config),
                            configureModel(config),
                            modelFilename,
                            config
                    ));
        }

        VehicleDetectionSystem.log.info("----- Model Loaded -----");

        return model;
    }

    private ComputationGraph loadTrainedModel(String modelFilename) throws Exception {
        VehicleDetectionSystem.log.info("Load model...");
        return ModelSerializer.restoreComputationGraph(modelFilename);
    }

    private Dataset loadDataset(File imageDir, ModelConfig config) throws Exception {
        VehicleDetectionSystem.log.info("Load dataset...");

        RandomPathFilter pathFilter = new RandomPathFilter(config.rng);
        InputSplit[] data = new FileSplit(imageDir, NativeImageLoader.ALLOWED_FORMATS, config.rng).sample(pathFilter, 0.9, 0.1);
        InputSplit trainData = data[0];
        InputSplit testData = data[1];

        ObjectDetectionRecordReader recordReaderTrain = new ObjectDetectionRecordReader(config.height, config.width, config.nChannels, config.gridHeight, config.gridWidth,
                new LabelImgXmlLabelProvider(imageDir));
        recordReaderTrain.initialize(trainData);

        ObjectDetectionRecordReader recordReaderTest = new ObjectDetectionRecordReader(config.height, config.width, config.nChannels, config.gridHeight, config.gridWidth,
                new LabelImgXmlLabelProvider(imageDir));
        recordReaderTest.initialize(testData);

        RecordReaderDataSetIterator train = new RecordReaderDataSetIterator(recordReaderTrain, config.batchSize, 1, 1, true);
        train.setPreProcessor(new ImagePreProcessingScaler(0, 1));
        RecordReaderDataSetIterator test = new RecordReaderDataSetIterator(recordReaderTest, 1, 1, 1, true);
        test.setPreProcessor(new ImagePreProcessingScaler(0, 1));

        Dataset dataset = new Dataset();
        dataset.train = train;
        dataset.test = test;

        return dataset;
    }

    private ComputationGraph configureModel(ModelConfig config) throws Exception {
        // YOLO2
        ZooModel zooModel = YOLO2.builder().build();
        ComputationGraph yolo2 = (ComputationGraph) zooModel.initPretrained();
        INDArray priors = Nd4j.create(config.priorBoxes);

        VehicleDetectionSystem.log.info(yolo2.summary());

        // Override the setting for all layers that are not "frozen".
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .seed(config.seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(1.0)
                .updater(new Adam.Builder().learningRate(config.learningRate).build())
                .l2(0.00001)
                .activation(Activation.IDENTITY)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .build();

        //Construct a new model with the intended architecture and print summary
        ComputationGraph yoloTransfer = new TransferLearning.GraphBuilder(yolo2)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor("leaky_re_lu_18") // Freeze layers
                .removeVertexKeepConnections("conv2d_23")
                .removeVertexKeepConnections("outputs")
                .addLayer("conv2d_23",
                        new ConvolutionLayer.Builder(1, 1)
                                .nIn(1024)
                                .nOut(config.nBoxes * (5 + config.nClasses))
                                .stride(1, 1)
                                .convolutionMode(ConvolutionMode.Same)
                                .weightInit(WeightInit.XAVIER)
                                .activation(Activation.IDENTITY)
                                .build(),
                        "leaky_re_lu_22")
                .addLayer("outputs",
                        new Yolo2OutputLayer.Builder()
                                .lambdaNoObj(config.lambdaNoObj)
                                .lambdaCoord(config.lambdaCoord)
                                .boundingBoxPriors(priors.castTo(DataType.FLOAT))
                                .build(),
                        "conv2d_23")
                .setOutputs("outputs")
                .build();
        VehicleDetectionSystem.log.info(yoloTransfer.summary());

        return yoloTransfer;
    }

    private String trainModel(Dataset dataset, ComputationGraph yoloTransfer, String modelFilename, ModelConfig config) throws Exception {
        VehicleDetectionSystem.log.info("Train model...");
        UIServer server = UIServer.getInstance();
        StatsStorage storage = new InMemoryStatsStorage();
        server.attach(storage);
        yoloTransfer.setListeners(new ScoreIterationListener(1), new StatsListener(storage));

        for (int i = 0; i < config.nEpochs; i++) {
            dataset.train.reset();
            VehicleDetectionSystem.log.info("----- Starting epoch {} -----", i);
            while (dataset.train.hasNext()) {
                yoloTransfer.fit(dataset.train.next());
            }
            VehicleDetectionSystem.log.info("----- Completed epoch {} -----", i);
            // Save after each 5 epoch
            if ((i + 1) % 5 == 0) {
                String checkpointFilename = "Model4_" + (i + 1) + ".zip";
                ModelSerializer.writeModel(yoloTransfer, checkpointFilename, true);
                VehicleDetectionSystem.log.info("----- Checkpoint Saved -----");
            }
            // Use this to pause after every epoch
//                System.out.println("Press enter to continue");
//                try{
//                    System.in.read();
//                }catch(Exception ignored){}
        }

        ModelSerializer.writeModel(yoloTransfer, modelFilename, true);
        VehicleDetectionSystem.log.info("Final Model Saved");

        return modelFilename;
    }
}
