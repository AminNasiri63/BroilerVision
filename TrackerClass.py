from abc import ABC, abstractmethod
from typing import List
import numpy as np
from DetectionClass import Detection
from deep_sort.deep_sort import Deep



class BaseTracker(ABC):
    """Abstract interface for any multi-object tracker."""

    @abstractmethod
    def update(self, detect_info) -> List[Detection]:
        """
        Update tracker with current frame detections and
        return detections with track_id filled.
        """
        pass


class DeepSortTracker(BaseTracker):
    """
    Wrapper for DeepSORT.
    """

    def __init__(self):
        self._init_tracker()

    def _init_tracker(self):
        """Initialize DeepSORT."""
        self.deep = Deep(max_distance=0.7,
            nms_max_overlap=1,
            n_init=3,
            max_age=15,
            max_iou_distance=0.7)
        
        self.tracker = self.deep.sort_tracker()

        print(f"[DeepSortTracker] Initialized")

    def update(self, detect_info) -> List[Detection]:

        frame, boxes, scores, class_ids, class_names = detect_info

        dummy_detections: List[Detection] = []

        # Convert boxes for DeepSORT (xmin, ymin, width, height)
        deepsortBoxes = np.array(boxes)
        deepsortBoxes[:, 2] = deepsortBoxes[:, 2] - deepsortBoxes[:, 0]  # width = x2 - x1
        deepsortBoxes[:, 3] = deepsortBoxes[:, 3] - deepsortBoxes[:, 1]  # height = y2 - y1

        
        features = self.deep.encoder(frame, deepsortBoxes)
        detect = self.deep.Detection(deepsortBoxes, scores, class_ids, features)
        self.tracker.predict()
        (class_ids, object_ids, boxes, _, _) = self.tracker.update(detect)


        for bbox, score, class_id, class_name, object_id in zip(boxes, scores, class_ids, class_names, object_ids):
            bbox_det = Detection(bbox=bbox, score=score, class_id=class_id, class_name=class_name, track_id=object_id)

            dummy_detections.append(bbox_det)

        
        return dummy_detections