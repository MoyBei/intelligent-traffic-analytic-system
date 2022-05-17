package ai.certifai;

import org.bytedeco.opencv.opencv_core.Point;

public class DetectedObjectInfo {
    public Point objectCenterPoint;
    public Point objectTopLeftXY;
    public Point objectBottomRightXY;
    public String objectLabel;
    public String objectCurrentLabel;
    public int objectLostCount;


    public DetectedObjectInfo(Point pt){
        this.objectCenterPoint = pt;
        this.objectLostCount = 0;
    }

    public DetectedObjectInfo(Point objCenterPoint, String label, Point topLeftXY, Point bottomRightXY){
        this.objectCenterPoint = objCenterPoint;
        this.objectLabel = label;
        this.objectCurrentLabel = label;
        this.objectTopLeftXY = topLeftXY;
        this.objectBottomRightXY = bottomRightXY;
        this.objectLostCount = 0;
    }

    public void updatePoint(Point pt, Point topLeftXY, Point bottomRightXY){
        this.objectCenterPoint = pt;
        this.objectTopLeftXY = topLeftXY;
        this.objectBottomRightXY = bottomRightXY;
        this.objectLostCount = 0;
    }

    public void updateCurrentLabel(String currentLabel) {
        this.objectCurrentLabel = currentLabel;
    }

    public void increaseLostCount(){
        this.objectLostCount += 1;
    }


}
