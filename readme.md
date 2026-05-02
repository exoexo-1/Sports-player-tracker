# 🏃 Multi-Object Detection & Persistent ID Tracking in Sports Footage

> **Assignment:** AI / Computer Vision / Data Science
> **Objective:** Detect, track, and maintain persistent identities of multiple subjects in real-world sports/event videos.

---

## 🎯 Objective

This project implements a complete computer vision pipeline capable of:

* Detecting multiple moving subjects (players, ball)
* Assigning **unique and persistent IDs**
* Handling real-world challenges:

  * occlusion
  * motion blur
  * camera motion (pan/zoom)
  * similar-looking players
* Producing:

  * annotated output video
  * heatmaps
  * trajectory visualizations
  * statistics

---

## 🧠 Approach Overview

```
Input Video
     ↓
YOLOv8 (Detection)
     ↓
BoT-SORT + ReID (Tracking)
     ↓
ID Stabilization Layer
     ↓
Outputs (Video + Analytics)
```

---

## ⚙️ Tech Stack

* **Detection:** YOLOv8x (Ultralytics)
* **Tracking:** BoT-SORT
* **Re-Identification:** OSNet embeddings
* **Backend:** FastAPI
* **Frontend:** HTML + JS (Vercel deployed)
* **Visualization:** OpenCV + Matplotlib

---

## 📂 Project Structure

```
Sports-player-tracker/
│
├── deploy/
│   ├── api.py                # FastAPI backend
│   ├── pipeline.py          # Core CV pipeline
│   ├── index.html           # Frontend UI
│   ├── requirements.txt     # Backend dependencies
│   ├── outputs/             # Generated outputs (ignored in Git)
│   └── uploads/             # Uploaded videos (runtime only)
│
├── main.ipynb               # Notebook version (development)
├── botsort_custom.yaml      # Tracker configuration
├── requirements.txt         # Root dependencies
├── readme.md                # Documentation
├── technical_report.docx    # Technical report (mandatory)
│
├── video_football.mp4       # Sample input (not used in deployment)
├── video_rugby.mp4
└── video.mp4
```

---

## 🚀 How to Run (Local)

### 1. Clone repo

```bash
git clone https://github.com/exoexo-1/sports-player-tracker.git
cd sports-player-tracker/deploy
```

---

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Run backend

```bash
uvicorn api:app --reload
```

---

### 5. Open frontend

Open:

```
deploy/index.html
```

Upload a video → click **Run Detection & Tracking**

---

## 📦 Outputs

For each input video:

| Output                  | Description              |
| ----------------------- | ------------------------ |
| `*_tracked.mp4`         | Annotated video with IDs |
| `*_heatmap.png`         | Movement density heatmap |
| `*_trajectories.png`    | Player movement paths    |
| `*_ball_trajectory.png` | Ball movement            |
| `*_count_over_time.png` | Player count graph       |
| `*_stats.json`          | Full statistics          |

---

## 🔍 Key Features

* Persistent ID tracking across occlusions
* Re-identification using appearance embeddings
* Motion trails per player
* Heatmap visualization
* Object count analytics
* Ball tracking (COCO class 32)

---

## 🧠 ID Consistency Strategy

To maintain stable IDs:

1. Kalman Filter → motion prediction
2. ByteTrack matching → recovery after occlusion
3. Global Motion Compensation → camera movement correction
4. ReID embeddings → appearance matching
5. Custom ID Stabilizer → removes residual switches

---

## ⚠️ Assumptions

* Players belong to COCO class `person`
* Ball detection is less reliable due to size & motion
* Input video is public and contains multiple moving subjects

---

## 🚫 Limitations

* ID switches may occur for identical players
* Long occlusion (>3 sec) may cause ID reset
* CPU inference is slow (~1–3 FPS)
* Ball detection inconsistent in fast scenes

---

## 🔮 Improvements

* Fine-tune on sports dataset (SoccerNet, SportsMOT)
* TensorRT acceleration
* Jersey number OCR
* Bird’s-eye projection
* Real-time streaming support

---

## 📄 Technical Report

See:

```
technical_report.docx
```

Includes:

* model selection reasoning
* tracking logic
* challenges
* failure cases
* improvements

---

## 🎥 Demo Requirements (Assignment)

Included:

* Annotated output video ✔
* Sample screenshots ✔
* Code repository ✔
* README ✔
* Technical report ✔

---

## 🔗 Video Source

Example:

* Football video:
  https://www.istockphoto.com/video/teenage-boys-playying-soccer-on-sunny-day-outdoors

(Other videos are also publicly sourced)

---

## 💡 Notes

* Outputs and uploads are ignored in Git
* Large files (videos) are not stored in repo
* Designed for **offline processing**, not real-time

---

## 🏁 Conclusion

This project demonstrates a complete **multi-object tracking system** capable of handling real-world sports footage with persistent ID assignment and advanced analytics.

---
