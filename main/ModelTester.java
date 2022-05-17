package ai.certifai;

import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_videoio.VideoWriter;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ColorConversionTransform;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.YoloUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.awt.event.KeyEvent;
import java.io.File;
import java.io.IOException;
import java.util.*;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class ModelTester {
    public void testOneImage(String absolutePath, String outputPath, ComputationGraph yoloTransfer, ModelConfig config, boolean renderBBox) throws IOException {
        File file = new File(absolutePath);
        VehicleDetectionSystem.log.info("You are using this image file located at " +  absolutePath);


        NativeImageLoader nil = new NativeImageLoader(416, 416, 3);
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer yout = (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer) yoloTransfer.getOutputLayer(0);

        INDArray image = nil.asMatrix(file);
        scaler.transform(image);

        Mat inputImageMat = imread(absolutePath);
        int w = inputImageMat.cols();
        int h = inputImageMat.rows();
        INDArray outputs = yoloTransfer.outputSingle(image);
        List<DetectedObject> detectedObject = yout.getPredictedObjects(outputs, config.detectionThreshold);
        YoloUtils.nms(detectedObject, 0.3);

        for (DetectedObject obj : detectedObject) {
            double[] xy1 = obj.getTopLeftXY();
            double[] xy2 = obj.getBottomRightXY();
            Point objCenterPoint = new Point((int) Math.round(obj.getCenterX() * w / config.gridWidth), (int) Math.round(obj.getCenterY() * h / config.gridHeight));
            String objectLabel = config.labels.get(obj.getPredictedClass());
            double objConfidence = obj.getConfidence();

            Scalar objColor = null;
            if (Objects.equals(objectLabel, "car")) {
                objColor = new Scalar(124, 174, 54, 255);
            } else if (Objects.equals(objectLabel, "motorbike")) {
                objColor = new Scalar(152, 116, 24, 255);
            } else if (Objects.equals(objectLabel, "commercial")) {
                objColor = new Scalar(35, 217, 249, 255);
            }

            int x1 = (int) Math.round(w * xy1[0] / config.gridWidth);
            int y1 = (int) Math.round(h * xy1[1] / config.gridHeight);
            int x2 = (int) Math.round(w * xy2[0] / config.gridWidth);
            int y2 = (int) Math.round(h * xy2[1] / config.gridHeight);

            String textToPut = objectLabel.toUpperCase();
            int[] baseline = {0};
            Size textSize = getTextSize(textToPut, FONT_HERSHEY_PLAIN, 2, 2, baseline);

            if (renderBBox) {
                rectangle(inputImageMat, new Point(x1, y1), new Point(x2, y2), objColor, 2, 0, 0);
                rectangle(inputImageMat, new Point(x1, y1), new Point(x1 + 3 + textSize.get(0), y1 - 3 - textSize.get(1)), Scalar.BLACK, -1, 0, 0);
                putText(inputImageMat, textToPut, new Point(x1, y1), FONT_HERSHEY_PLAIN, 2, objColor, 2, 0, false);
            } else {
                rectangle(inputImageMat, objCenterPoint, new Point(objCenterPoint.x() + 3 + textSize.get(0), objCenterPoint.y() - 3 - textSize.get(1)), Scalar.BLACK, -1, 0, 0);
                putText(inputImageMat, textToPut, objCenterPoint, FONT_HERSHEY_PLAIN, 2, objColor, 2, 0, false);
            }

        }
        imwrite(outputPath, inputImageMat);
        VehicleDetectionSystem.log.info("Image Saved at: " + outputPath);
    }

    public void testVideo(String videoPath, String outputVideoPath, ComputationGraph yoloTransfer, ModelConfig config, boolean renderVideo, boolean renderBBox) throws Exception {
        FFmpegFrameGrabber grabber = new FFmpegFrameGrabber(videoPath);
        grabber.setFormat("mp4");
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        grabber.start();

        NativeImageLoader loader = new NativeImageLoader(416, 416, 3, new ColorConversionTransform(COLOR_BGR2RGB));
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer yout = (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer) yoloTransfer.getOutputLayer(0);

        VehicleDetectionSystem.log.info("----- Loading video -----");

        int i = 0;
        Frame frame = grabber.grab();
        Map<Integer, DetectedObjectInfo> tracking_objects = new HashMap<Integer, DetectedObjectInfo>();
        Map<Integer, String> roi_1_objects = new HashMap<Integer, String>();
        Map<Integer, String> roi_2_objects = new HashMap<Integer, String>();

        int w = frame.imageWidth;
        int h = frame.imageHeight;

        VideoWriter vidWriter = new VideoWriter(
                outputVideoPath,
                VideoWriter.fourcc((byte) 'm', (byte) 'p', (byte) '4', (byte) '2'),
                grabber.getFrameRate(),
                new Size(w, h));

        String winName = "Object Detection";
        CanvasFrame canvas = new CanvasFrame(winName);
        canvas.setCanvasSize(w, h);

        if (renderVideo){
            canvas.dispose();
        }


        int track_id = 0;


        while (frame != null) {
//            Map<Point, String> centerPointCurrentFrame = new HashMap<Point, String>();
            List<DetectedObjectInfo> centerPointCurrentFrame = new ArrayList<DetectedObjectInfo>();
            //if a thread is null, create new thread

            Mat rawImage = converter.convert(frame);
            Mat resizeImage = new Mat();
            resize(rawImage, resizeImage, new Size(config.width, config.height));
            INDArray inputImage = loader.asMatrix(resizeImage);
            scaler.transform(inputImage);

            INDArray outputs = yoloTransfer.outputSingle(inputImage);
            List<DetectedObject> detectedObject = yout.getPredictedObjects(outputs, config.detectionThreshold);
            YoloUtils.nms(detectedObject, 0.1);
//            System.out.println(detectedObject);

            // Put detected object into list for tracking
            for (DetectedObject obj : detectedObject) {
                // Ignore center point at top area (clustered, low accuracy)
                if ((int) Math.round(obj.getCenterY() * h / config.gridHeight) > 350) {
                    double[] xy1 = obj.getTopLeftXY();
                    Point topLeftXY = new Point((int) Math.round(w * xy1[0] / config.gridWidth), (int) Math.round(h * xy1[1] / config.gridHeight));

                    double[] xy2 = obj.getBottomRightXY();
                    Point bottomRightXY = new Point((int) Math.round(w * xy2[0] / config.gridWidth), (int) Math.round(h * xy2[1] / config.gridHeight));

                    String label = config.labels.get(obj.getPredictedClass());
                    double objConfidence = obj.getConfidence();
                    Point objCenterPoint = new Point((int) Math.round(obj.getCenterX() * w / config.gridWidth), (int) Math.round(obj.getCenterY() * h / config.gridHeight));

                    centerPointCurrentFrame.add(new DetectedObjectInfo(objCenterPoint, label, topLeftXY, bottomRightXY));
                }
            }

            // ----- Tracking Object Start -----
            if (i == 0) {
                for (DetectedObjectInfo object : centerPointCurrentFrame) {
                    Point pt = object.objectCenterPoint;
                    String objLabel = object.objectLabel;
                    Point objTopLeftXY = object.objectTopLeftXY;
                    Point objBottomRightXY = object.objectBottomRightXY;

                    tracking_objects.put(track_id, new DetectedObjectInfo(pt, objLabel, objTopLeftXY, objBottomRightXY));
                    track_id += 1;
                }
            } else {
                Map<Integer, DetectedObjectInfo> tracking_objects_copy = new HashMap<Integer, DetectedObjectInfo>();
                for (Map.Entry<Integer, DetectedObjectInfo> entry : tracking_objects.entrySet()) {
                    Integer objectID = entry.getKey();
                    DetectedObjectInfo objectPoint = entry.getValue();
                    tracking_objects_copy.put(objectID, objectPoint);
                }


                for (Map.Entry<Integer, DetectedObjectInfo> entry : tracking_objects_copy.entrySet()) {
                    Integer objectID = entry.getKey();
                    Point pt2 = entry.getValue().objectCenterPoint;
                    boolean object_exists = false;

                    List<DetectedObjectInfo> centerPointCurrentFrame_copy = new ArrayList<DetectedObjectInfo>(centerPointCurrentFrame);

                    for (DetectedObjectInfo object : centerPointCurrentFrame_copy) {
                        Point pt = object.objectCenterPoint;
                        String objLabel = object.objectLabel;
                        Point objTopLeftXY = object.objectTopLeftXY;
                        Point objBottomRightXY = object.objectBottomRightXY;
                        double distance = Math.hypot(pt2.x() - pt.x(), pt2.y() - pt.y());

                        //Update IDs Info
                        if (distance < 80) {
                            DetectedObjectInfo current_object = tracking_objects.get(objectID);
                            current_object.updatePoint(pt, objTopLeftXY, objBottomRightXY);
                            current_object.updateCurrentLabel(objLabel);
                            object_exists = true;

                            // Remove the object
                            centerPointCurrentFrame.remove(object);
                        }

                    }
                    // Remove IDs lost
                    if (!object_exists) {
                        tracking_objects.get(objectID).increaseLostCount();
                        if (tracking_objects.get(objectID).objectLostCount > 5) {
                            tracking_objects.remove(objectID);
                        }
                    }

                }
                // Add new IDs found
                for (DetectedObjectInfo object : centerPointCurrentFrame) {
                    Point pt = object.objectCenterPoint;
                    String objLabel = object.objectLabel;
                    Point objTopLeftXY = object.objectTopLeftXY;
                    Point objBottomRightXY = object.objectBottomRightXY;

                    boolean valueExisted = false;
                    for (Map.Entry<Integer, DetectedObjectInfo> entry : tracking_objects_copy.entrySet()) {
                        if (pt.equals(entry.getValue().objectCenterPoint)) {
                            valueExisted = true;
                            break;
                        }
                    }
                    // Prevent new ID to same point
                    if (!valueExisted) {
                        tracking_objects.put(track_id, new DetectedObjectInfo(pt, objLabel, objTopLeftXY, objBottomRightXY));
                        track_id += 1;
                    }
                }

            }

            // Draw Point and Label
            for (Map.Entry<Integer, DetectedObjectInfo> entry : tracking_objects.entrySet()) {
                int objectID = entry.getKey();
                Point objectPoint = entry.getValue().objectCenterPoint;
                String objectLabel = entry.getValue().objectCurrentLabel;
                int objectLostCounter = entry.getValue().objectLostCount;
                Point objectTopLeftXY = entry.getValue().objectTopLeftXY;
                Point objectBottomRightXY = entry.getValue().objectBottomRightXY;


                Scalar objColor = null;
                if (Objects.equals(objectLabel, "car")) {
                    objColor = new Scalar(124, 174, 54, 255);
                } else if (Objects.equals(objectLabel, "motorbike")) {
                    objColor = new Scalar(152, 116, 24, 255);
                } else if (Objects.equals(objectLabel, "commercial")) {
                    objColor = new Scalar(35, 217, 249, 255);
                }

                // Bounding Box
                if (renderBBox && objectLostCounter == 0) {
                    rectangle(rawImage, objectTopLeftXY, objectBottomRightXY, objColor, 2, 0, 0);
                }

                String textToPut = objectID + "|" + objectLabel.toUpperCase();
                int[] baseline = {0};
                Size textSize = getTextSize(textToPut, FONT_HERSHEY_PLAIN, 2, 2, baseline);
                rectangle(rawImage, objectPoint, new Point(objectPoint.x() + 3 + textSize.get(0), objectPoint.y() - 3 - textSize.get(1)), Scalar.BLACK, -1, 0, 0);
                putText(rawImage, textToPut, objectPoint, FONT_HERSHEY_PLAIN, 2, objColor, 2, 0, false);
            }

            // ----- Tracking Object End -----

            // ----- Object Counting Start -----
            int roi_1_TopLeftX = 1080;
            int roi_1_TopLeftY = 700;
            int roi_1_BottomRightX = 1900;
            int roi_1_BottomRightY = 780;

            Point roi_1_TopLeftXY = new Point(roi_1_TopLeftX, roi_1_TopLeftY);
            Point roi_1_BottomRightXY = new Point(roi_1_BottomRightX, roi_1_BottomRightY);

            int roi_2_TopLeftX = 470;
            int roi_2_TopLeftY = 435;
            int roi_2_BottomRightX = 1110;
            int roi_2_BottomRightY = 515;

            Point roi_2_TopLeftXY = new Point(roi_2_TopLeftX, roi_2_TopLeftY);
            Point roi_2_BottomRightXY = new Point(roi_2_BottomRightX, roi_2_BottomRightY);

            rectangle(rawImage, roi_1_TopLeftXY, roi_1_BottomRightXY, Scalar.YELLOW, 2, 0, 0);
            putText(rawImage, "ROI_1", roi_1_TopLeftXY, FONT_HERSHEY_PLAIN, 2, Scalar.YELLOW, 2, 0, false);
            rectangle(rawImage, roi_2_TopLeftXY, roi_2_BottomRightXY, Scalar.YELLOW, 2, 0, 0);
            putText(rawImage, "ROI_2", roi_2_TopLeftXY, FONT_HERSHEY_PLAIN, 2, Scalar.YELLOW, 2, 0, false);


            for (Map.Entry<Integer, DetectedObjectInfo> entry : tracking_objects.entrySet()) {
                Integer objectID = entry.getKey();
                Point objectPoint = entry.getValue().objectCenterPoint;
                String objLabel = entry.getValue().objectCurrentLabel;

                // Check if objectPoint isInBox1
                boolean isInBox1 = objectPoint.x() > roi_1_TopLeftX &&
                        objectPoint.x() < roi_1_BottomRightX &&
                        objectPoint.y() > roi_1_TopLeftY &&
                        objectPoint.y() < roi_1_BottomRightY;

                // If ID in box and not in List, add ID into list
                if (isInBox1 && !roi_1_objects.containsKey(objectID)) {
                    roi_1_objects.put(objectID, objLabel);
                    System.out.println(objectID + " is in the box.");
                }

                // Check if objectPoint isInBox2
                boolean isInBox2 = objectPoint.x() > roi_2_TopLeftX &&
                        objectPoint.x() < roi_2_BottomRightX &&
                        objectPoint.y() > roi_2_TopLeftY &&
                        objectPoint.y() < roi_2_BottomRightY;

                // If ID in box and not in List, add ID into list
                if (isInBox2 && !roi_2_objects.containsKey(objectID)) {
                    roi_2_objects.put(objectID, objLabel);
                    System.out.println(objectID + " is in the box.");
                }
            }

            // Print text box 1
            int car_count_1 = (int) roi_1_objects.values().stream().filter(v -> Objects.equals(v, "car")).count();
            int motorbike_count_1 = (int) roi_1_objects.values().stream().filter(v -> Objects.equals(v, "motorbike")).count();
            int commercial_count_1 = (int) roi_1_objects.values().stream().filter(v -> Objects.equals(v, "commercial")).count();

            String textToPut1 = "ROI_1->Car:" + car_count_1 + " Motorbike:" + motorbike_count_1 + " Commercial:" + commercial_count_1;
            int[] baseline1 = {0};
            Size textSize1 = getTextSize(textToPut1, FONT_HERSHEY_DUPLEX, 1, 1, baseline1);
            rectangle(rawImage, new Point(0, 0), new Point(textSize1.get(0) + 10, textSize1.get(1) + 15), Scalar.BLACK, -1, 0, 0);

            Scalar textColor1 = new Scalar(83, 83, 235, 255);
            putText(rawImage, textToPut1, new Point(5, 30), FONT_HERSHEY_DUPLEX, 1, textColor1);

            // Print text box 2
            int car_count_2 = (int) roi_2_objects.values().stream().filter(v -> Objects.equals(v, "car")).count();
            int motorbike_count_2 = (int) roi_2_objects.values().stream().filter(v -> Objects.equals(v, "motorbike")).count();
            int commercial_count_2 = (int) roi_2_objects.values().stream().filter(v -> Objects.equals(v, "commercial")).count();

            String textToPut2 = "ROI_2->Car:" + car_count_2 + " Motorbike:" + motorbike_count_2 + " Commercial:" + commercial_count_2;
            int[] baseline2 = {0};
            Size textSize2 = getTextSize(textToPut2, FONT_HERSHEY_DUPLEX, 1, 1, baseline2);
            rectangle(rawImage, new Point(0, 35), new Point(textSize2.get(0) + 10, 35 + textSize2.get(1) + 15), Scalar.BLACK, -1, 0, 0);

            Scalar textColor2 = new Scalar(83, 83, 235, 255);
            putText(rawImage, textToPut2, new Point(5, 65), FONT_HERSHEY_DUPLEX, 1, textColor2);


            // ----- Object Counting End -----


            // ----- Video Renderer -----
            if (renderVideo) {
                vidWriter.write(rawImage);
                VehicleDetectionSystem.log.info("----- Writing frame {} -----", i);
            } else {
                canvas.showImage(converter.convert(rawImage));
                KeyEvent t = canvas.waitKey(0);

                if ((t != null) && (t.getKeyCode() == KeyEvent.VK_ESCAPE)) {
                    break;
                }
            }

            frame = grabber.grab();
            i++;

        }

        if (renderVideo) {
            vidWriter.release();
            VehicleDetectionSystem.log.info("-----Finished Rendering-----");
        } else {
            canvas.dispose();
        }
    }
}
