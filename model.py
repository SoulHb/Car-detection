import os
import numpy as np
import torch
import torchvision
import cv2
from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from PIL import Image
from dataset import DetectDataset
from torchvision import transforms as T


def faster_rcnn(num_classes=2):
    f_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = f_rcnn.roi_heads.box_predictor.cls_score.in_features
    f_rcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return f_rcnn



