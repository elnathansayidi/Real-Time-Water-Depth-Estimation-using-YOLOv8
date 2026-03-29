# 💧 Real-Time Water Depth Estimation using YOLOv8

A computer vision system that estimates real-world water depth from video using segmentation and homography.

---

## 🚀 Features

* YOLOv8 segmentation for water detection
* Real-time waterline detection
* Depth estimation using homography
* Interactive calibration
* Optional OCR validation

---

## 🧠 Pipeline

Video → Segmentation → Water Mask → Waterline → Homography → Depth (meters)

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Run

```bash
python water_depth_detector.py --source sample_videos/demo.mp4 --model best.pt
```

---

## 📸 Output

* Waterline overlay
* Depth in meters
* Stable predictions

---

## 🎥 Demo Video

▶️ [Watch full demo](result/resultvid.mp4)

## ⚠️ Notes

* Calibration required before first run
* OCR is optional
* Works best with clear gauge

---

## 👨‍💻 Author

Annel Jefferson S A 
