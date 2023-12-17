import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def faster_rcnn(num_classes=2):
    """
        Create a Faster R-CNN model with a ResNet-50 backbone and FPN architecture.

        Args:
            num_classes (int): Number of output classes. Default is 2.

        Returns:
            torchvision.models.detection.FasterRCNN: Faster R-CNN model.
        """
    f_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = f_rcnn.roi_heads.box_predictor.cls_score.in_features
    f_rcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return f_rcnn



