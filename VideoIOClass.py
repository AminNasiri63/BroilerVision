
from typing import Tuple
import cv2
import numpy as np


class VideoReader:
    def __init__(self, source: str):

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

    def read(self) -> Tuple[bool, np.ndarray]:
        return self.cap.read()

    def release(self):
        self.cap.release()


class VideoWriter:
    def __init__(self, output_path: str, fps: float, frame_size: Tuple[int, int]):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    def write(self, frame: np.ndarray):
        self.writer.write(frame)

    def release(self):
        self.writer.release()