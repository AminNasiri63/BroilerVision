from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]  # x1, y1, w, h
    score: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None   # filled by tracker