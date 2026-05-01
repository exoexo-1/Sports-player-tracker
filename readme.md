# 🏃 Multi-Object Detection & Persistent ID Tracking

> **YOLOv8x + BoT-SORT + OSNet Re-ID** pipeline for detecting and persistently tracking players (and the ball) in sports footage — with movement heatmaps, trajectory visualization, and per-player coloured trails.

---

## 📽️ Demo

| Annotated Output | Movement Heatmap | Trajectories |
|:---:|:---:|:---:|
| Coloured bounding boxes + persistent IDs + trails | Player density across the pitch | Movement paths for top 20 players |

---

## ✨ Features

- **Persistent player IDs** — IDs survive occlusion, overlap, and re-entry using a 4-layer defence stack
- **Ball detection** — separate low-confidence tracking for sports ball (COCO class 32)
- **Per-player colour coding** — each player gets a unique, consistent colour derived from their ID
- **Trajectory trails** — fading motion trails drawn behind each player
- **Movement heatmap** — Gaussian-blurred density map overlaid on the field
- **Object count over time** — time-series plot of active player count per frame
- **Ball trajectory plot** — ball movement path coloured by time
- **Model comparison** — YOLOv8x vs YOLOv9e detection count comparison chart
- **Auto-named outputs** — all output files named after the input video (no overwriting between runs)
- **Full stats JSON** — machine-readable summary of every run

---

## 🏗️ Architecture

```
Input Video
     │
     ▼
┌─────────────────────────────┐
│   YOLOv8x  (imgsz=1280)     │  ← Detects persons + sports ball
│   conf=0.35 / 0.25 (ball)   │
└────────────┬────────────────┘
             │ detections
             ▼
┌─────────────────────────────┐
│   BoT-SORT Tracker          │
│   ├─ Kalman Filter          │  ← Motion prediction
│   ├─ ByteTrack dual match   │  ← Occlusion recovery
│   ├─ GMC sparseOptFlow      │  ← Camera motion correction
│   └─ OSNet Re-ID (512-dim)  │  ← Appearance-based re-entry match
└────────────┬────────────────┘
             │ raw track IDs
             ▼
┌─────────────────────────────┐
│   IDStabilizer              │  ← Spatial post-processing re-match
│   (100px / 3s window)       │     eliminates residual ID switches
└────────────┬────────────────┘
             │ stable IDs
             ▼
┌─────────────────────────────┐
│   Annotated Output Video    │
│   + Heatmap + Trajectories  │
│   + Stats JSON              │
└─────────────────────────────┘
```

---

## 📦 Installation

### Requirements
- Python 3.10+
- CUDA-capable GPU (NVIDIA recommended — tested on L4)
- ~3 GB VRAM for YOLOv8x at imgsz=1280

### Install dependencies

```bash
pip install "ultralytics==8.3.57" lapx supervision boxmot
```

> **Why pin ultralytics to 8.3.57?**  
> Ultralytics 8.4+ changed the BoT-SORT YAML config schema in a breaking way. The pinned version keeps the custom tracker config compatible.

---

## 🚀 Usage

### 1. Open `main.ipynb` in Lightning.ai (or Jupyter)

### 2. Set your input video — edit **one line** in Cell 4:

```python
INPUT_VIDEO = "video_football.mp4"   # ← change this
```

### 3. Run All Cells (`Kernel → Run All`)

All outputs are automatically named after your input video:

| Input | Outputs |
|-------|---------|
| `video_football.mp4` | `video_football_tracked.mp4` |
| | `video_football_heatmap.png` |
| | `video_football_trajectories.png` |
| | `video_football_ball_trajectory.png` |
| | `video_football_count_over_time.png` |
| | `video_football_model_comparison.png` |
| | `video_football_stats.json` |

---

## 📂 Output Files

| File | Description |
|------|-------------|
| `*_tracked.mp4` | Annotated video — coloured bounding boxes, persistent IDs, motion trails, ball detection, HUD overlay |
| `*_heatmap.png` | Player movement density heatmap blended over a sample frame |
| `*_trajectories.png` | Static path plot for the 20 most active players |
| `*_ball_trajectory.png` | Ball movement path coloured by time (if ball detected) |
| `*_count_over_time.png` | Active tracked IDs per frame with 2-second rolling average |
| `*_model_comparison.png` | YOLOv8x vs YOLOv9e detection count comparison |
| `*_stats.json` | Full run statistics (IDs, fps, ball detection rate, config) |

---

## ⚙️ Configuration Reference

All tunables live in `CONFIG` (Cell 4). Nothing else needs changing.

```python
INPUT_VIDEO = "your_video.mp4"   # ← only thing to change per video

CONFIG = {
    # Detection
    "model_path"    : "yolov8x.pt",
    "conf_person"   : 0.35,      # lower = catch more occlusions
    "conf_ball"     : 0.25,      # lower = ball is small/fast
    "iou_thresh"    : 0.45,
    "imgsz"         : 1280,      # higher = better small-player detection

    # Tracker
    "track_buffer"      : 90,    # frames to keep lost track alive (~3s at 25fps)
    "appearance_thresh" : 0.25,  # cosine distance for Re-ID match (lower = stricter)
    "match_thresh"      : 0.80,  # IoU threshold for track-detection association

    # Visuals
    "max_trail_len" : 90,        # frames of trail per player
}
```

---

## 🔍 How ID Consistency Works

ID switching is the hardest problem in multi-object tracking. This pipeline uses **5 layers**:

| Layer | What it does | Problem it solves |
|-------|-------------|-------------------|
| Kalman Filter | Predicts next-frame position | Fast movement, short detection gaps |
| ByteTrack dual-threshold | Second-chance matching for low-conf detections | Partial occlusion |
| GMC (sparseOptFlow) | Corrects predictions for camera motion | Pan, zoom, shake |
| OSNet Re-ID | Appearance embedding similarity match | Player re-enters frame |
| IDStabilizer | Spatial proximity re-match (100px, 3s) | Residual switches |

---

## 📊 Example Results

```
Video     : video_rugby.mp4
Resolution: 3840×2160  |  FPS: 25  |  Frames: 358  |  Duration: 14.3s
Model     : YOLOv8x  |  Tracker: BoT-SORT + Re-ID

Stable unique IDs : 117   ← target: ~15–20 (ground truth players)
Peak simultaneous : 49
Avg simultaneous  : 32.8
Ball detected     : 13/358 frames (3.6%)
Processing speed  : 3.6 fps  (99s total on L4 GPU)
```

> Note: 117 IDs vs ~15 ground-truth is primarily from players exiting/re-entering over 3+ seconds (beyond the track buffer). See [Technical Report](technical_report.docx) for full analysis.

---

## ⚡ Performance Tips

| Goal | Setting |
|------|---------|
| Faster processing | Lower `imgsz` to `640` or `960` |
| Better small-player detection | Keep `imgsz=1280` or raise to `1536` |
| Fewer ID switches | Increase `track_buffer` (costs more memory) |
| Stricter Re-ID (less false re-matches) | Lower `appearance_thresh` toward `0.15` |
| More permissive Re-ID | Raise `appearance_thresh` toward `0.4` |
| Ball-only video | Set `target_classes: [32]` and `conf_ball: 0.15` |

---

## 🧠 Model Comparison

The optional Cell 15 runs **YOLOv8x vs YOLOv9e** on the first 300 frames and saves a comparison chart. From testing on sports footage:

| Model | Avg detections | Peak | Notes |
|-------|---------------|------|-------|
| YOLOv8x | ~35 | ~55 | Higher recall, more false positives on crowd |
| YOLOv9e | ~31 | ~45 | Fewer detections, slightly more precise |

YOLOv8x was chosen as the primary detector for its higher recall — missing a real player is worse than a brief false positive.

---

## 🐛 Known Limitations

- Ball detection is unreliable when the ball is motion-blurred or occluded (detected in <5% of rugby frames)
- Players absent from frame for >3 seconds receive a new ID on return
- Identical jerseys reduce Re-ID accuracy — same-team players crossing paths can swap IDs
- 4K footage at imgsz=1280 runs at ~3–4 fps (offline only; use TensorRT for faster inference)

---

## 🔮 Possible Improvements

- **Sport-specific fine-tuning** — train on SoccerNet / SportsMOT for better recall
- **TensorRT export** — `model.export(format='engine')` for 2–3× GPU speedup
- **Jersey Re-ID** — add colour histogram or jersey number OCR as extra features
- **Bird's-eye projection** — homography mapping to top-down pitch view for speed estimation
- **OC-SORT / StrongSORT** — better non-linear motion handling for tackles and direction changes
- **Temporal track merging** — post-process to retroactively merge short track fragments

---

## 📋 Dependencies

```
ultralytics==8.3.57
lapx
supervision
boxmot
opencv-python
torch (CUDA)
matplotlib
numpy
pyyaml
```

---

## 📄 License

For academic / assignment purposes. Video sources must be publicly accessible and credited in submissions.

---

## 🎥 Video Sources

| File | Source URL |
|------|-----------|
| `video_football.mp4` | *https://www.istockphoto.com/video/teenage-boys-playying-soccer-on-sunny-day-outdoors-gm2217984923-634547682?utm_source=pexels&utm_medium=affiliate&utm_campaign=sponsored_video&utm_content=srp_inline_portrait_media&utm_term=football
* |
| `video_rugby.mp4` | *forgot* |
| `video.mp4` | *forgot* |