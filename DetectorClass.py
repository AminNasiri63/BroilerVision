import keras_cv
from abc import ABC, abstractmethod
from typing import List
import numpy as np
import cv2
import tensorflow as tf


class BaseDetector(ABC):

    @abstractmethod
    def detect(self, frame: np.ndarray):
        """Return list of detections for a frame."""
        pass


class YoloDetector(BaseDetector):
    """
    Example YOLO wrapper.
    """
    objs_name = ["broiler"]
    score_thr = 0.05

    def __init__(self, model_path: str, cropBox, scaleImg=20):
        self.model_path = model_path
        self.cropBox = cropBox
        self.scaleImg = scaleImg
        self._load_model()

    def _load_model(self):
        """Load YOLO model here."""

        backbone = keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_xl_backbone_coco")
        self.yolo = keras_cv.models.YOLOV8Detector(
            num_classes=len(self.objs_name),
            bounding_box_format="xyxy",
            backbone=backbone,
            fpn_depth=1)
        self.yolo.load_weights(self.model_path)

        print(f"[YoloDetector] Model loaded from: {self.model_path}")

    
    def modifyBox(self, shape, boundingBox, pad=10):
        hImg, wImg, _ = shape

        (x1, y1, x2, y2) = boundingBox
        _wBox, _hBox = x2 - x1, y2 - y1

        side = max(_wBox, _hBox)

        centerX, centerY = (x1 + x2) // 2, (y1 + y2) // 2

        halfSide = side // 2
        x1 = centerX - halfSide
        x2 = centerX + halfSide
        y1 = centerY - halfSide
        y2 = centerY + halfSide

        pad = halfSide * (pad / (2 * 100))

        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(wImg - 1, x2 + pad), min(hImg - 1, y2 + pad)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        return x1, y1, x2, y2
    
    
    def detect(self, frame: np.ndarray):
        """
        Run YOLO inference on a frame and convert outputs to List[Detection].
        """
        target_size = [32*self.scaleImg, 32*self.scaleImg]
        width = self.cropBox[2] - self.cropBox[0]
        height = self.cropBox[3] - self.cropBox[1]
        fx = width / target_size[0]
        fy = height / target_size[0]
        scaleFactor = tf.constant([fx, fy, fx, fy], dtype=tf.float32)

        img = frame[self.cropBox[1]:self.cropBox[3], self.cropBox[0]:self.cropBox[2]]

        imgYOLO = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgYOLO = tf.cast(imgYOLO, tf.float32)
        imgYOLO = tf.image.resize(imgYOLO, target_size)
        imgYOLO = tf.expand_dims(imgYOLO, axis=0)

        y_pred = self.yolo.predict(imgYOLO, verbose=0)
        boxes = y_pred['boxes']
        scores = y_pred['confidence']
        class_ids = y_pred['classes']

        boxes = tf.squeeze(boxes, axis=0)
        scores = tf.squeeze(scores, axis=0)
        class_ids = tf.squeeze(class_ids, axis=0)
        boxes = tf.multiply(boxes, scaleFactor)

        scores, boxes, class_ids = scores.numpy(), boxes.numpy(), class_ids.numpy()
        boxes = boxes[scores > self.score_thr]
        boxes = np.array([self.modifyBox(img.shape, box) for box in boxes])
        class_ids = class_ids[scores > self.score_thr]
        scores = scores[scores > self.score_thr]

        class_names = [self.objs_name[i] for i in class_ids]

        return img, boxes, scores, class_ids, class_names
