from typing import List, Optional, Tuple
import cv2
import numpy as np
from DetectionClass import *
from TrackerClass import *
from VideoIOClass import *
from DetectorClass import *


class DetectionTrackingPipeline:

    def __init__(
        self,
        detector: BaseDetector,
        tracker: BaseTracker,
        reader: VideoReader,
        writer: Optional[VideoWriter] = None,
        show_window: bool = True,
    ):
        self.detector = detector
        self.tracker = tracker
        self.reader = reader
        self.writer = writer
        self.show_window = show_window

    def _draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            track_text = f"ID {det.track_id}" if det.track_id is not None else ""
            label = f"{det.class_name} {det.score:.2f} {track_text}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return frame

    def run(self):
        while True:
            ret, frame = self.reader.read()
            if not ret:
                break

            detect_info = self.detector.detect(frame)
            tracked_detections = self.tracker.update(detect_info)
            vis_frame = self._draw_detections(detect_info[0], tracked_detections)

            if self.writer is not None:
                self.writer.write(vis_frame)

            if self.show_window:
                cv2.imshow("Det + Track", vis_frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break

        self._cleanup()

    def _cleanup(self):
        self.reader.release()
        if self.writer is not None:
            self.writer.release()
        if self.show_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "./input.mp4"
    output_path = "./output_tracked.mp4"
    cropBox=[24, 855, 984, 1495]

    reader = VideoReader(video_path)

    cap_tmp = cv2.VideoCapture(video_path)
    fps = cap_tmp.get(cv2.CAP_PROP_FPS)
    w = cropBox[2] - cropBox[0]
    h = cropBox[3] - cropBox[1]
    cap_tmp.release()

    writer = VideoWriter(output_path, fps=fps, frame_size=(w, h))

    detector = YoloDetector(model_path="./Broiler20.h5", cropBox=cropBox)
    tracker = DeepSortTracker()

    pipeline = DetectionTrackingPipeline(
        detector=detector,
        tracker=tracker,
        reader=reader,
        writer=writer,
        show_window=True,
    )

    pipeline.run()