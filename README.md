## 🐔 BroilerVision
YOLOv8 + DeepSORT Pipeline for Broiler Detection and Tracking

BroilerVision is an end-to-end computer vision pipeline that reads a video, detects broiler chickens using a YOLO-based detector, and tracks them across frames using the DeepSORT multi-object tracking algorithm.

This system is designed for precision livestock farming, enabling automated monitoring of broiler behavior, movement patterns, and interactions in commercial poultry houses.

---

## 🔍 Features

✅ YOLO-based detection for fast, accurate broiler localization

✅ DeepSORT tracking for consistent ID assignment across frames

---

## 📦 Repository Structure

```
BroilerVision/
│   PipelineClass.py
│   DetectorClass.py
│   TrackerClass.py
│   DetectionClass.py
│   VideoIOClass.py
│   requirements.txt
│   README.md
│
└── deep_sort/

```
---

## 🎯 Model Weights

The trained YOLO model (Broiler20.h5, ~675 MB) is hosted on Hugging Face.

👉 Download here:
https://huggingface.co/AminNasiri63/YOLOToDetectBroilers

Use it programmatically:
```
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="AminNasiri63/YOLOToDetectBroilers",
    filename="Broiler.h5"
)

```

---

## 🛠 Installation
Clone the repository
```
git clone https://github.com/AminNasiri63/BroilerVision.git
cd BroilerVision

```
Install dependencies

```
pip install -r requirements.txt
```

----

## Download the YOLO model

Place the downloaded ```.h5``` model file inside the root folder:
```
BroilerVision/
│   Broiler.h5   <-- put here
│   ...

```

---

## ▶️ Usage
Run tracking on a video

Inside the script, simply modify the paths:
```
input_video = "input.mp4"
output_video = "output_tracked.mp4"
```

----




