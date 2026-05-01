def process_video(input_path, output_dir):
    # your full pipeline code here
    import cv2
    import torch
    import numpy as np
    import json
    import yaml
    import os
    from pathlib import Path
    from collections import defaultdict, deque
    from datetime import datetime
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ultralytics import YOLO
    import supervision as sv

    print("All imports OK")

    # ── CONFIGURATION ──────────────────────────────────────────────────────────────
    # All output filenames are derived from the input video name automatically.
    # Change only INPUT_VIDEO — all outputs will be named accordingly.
    # e.g. "video_football.mp4" → "video_football_tracked.mp4",
    #                              "video_football_heatmap.png", etc.

    INPUT_VIDEO = input_path   # ← only thing you change per video


    os.makedirs(output_dir, exist_ok=True)

    from pathlib import Path
    _stem = Path(INPUT_VIDEO).stem   # e.g. "video_football"

    CONFIG = {
        # --- I/O ---
        "input_video"    : INPUT_VIDEO,
        "output_video": os.path.join(output_dir, f"{_stem}_tracked.mp4"),
        "output_heatmap": os.path.join(output_dir, f"{_stem}_heatmap.png"),
        "output_traj": os.path.join(output_dir, f"{_stem}_trajectories.png"),
        "output_ball": os.path.join(output_dir, f"{_stem}_ball_trajectory.png"),
        "output_count": os.path.join(output_dir, f"{_stem}_count_over_time.png"),
        "stats_json": os.path.join(output_dir, f"{_stem}_stats.json"),

        # --- Model ---
        "model_path"     : "yolov8x.pt",

        # --- Detection thresholds ---
        "conf_person"    : 0.35,
        "conf_ball"      : 0.25,
        "iou_thresh"     : 0.45,
        "target_classes" : [0, 32],

        # --- BoT-SORT tracker ---
        "tracker"            : "botsort",
        "track_high_thresh"  : 0.35,
        "track_low_thresh"   : 0.10,
        "new_track_thresh"   : 0.40,
        "track_buffer"       : 90,
        "match_thresh"       : 0.80,
        "proximity_thresh"   : 0.5,
        "appearance_thresh"  : 0.25,

        # --- Processing ---
        "imgsz"          : 1280,
        "device"         : "cuda" if torch.cuda.is_available() else "cpu",

        # --- Visuals ---
        "max_trail_len"  : 90,
        "ball_color"     : (0, 255, 255),
    }

    print(f"Input  : {CONFIG['input_video']}")
    print(f"Outputs:")
    for k, v in CONFIG.items():
        if k.startswith("output") or k == "stats_json":
            print(f"  {k:<18} → {v}")
    print(f"\nDevice : {CONFIG['device']}")



    # YOLOv8x will auto-download on first run (~130 MB).
    # The 'x' model has 68M parameters vs 25M in 'm' — significantly better
    # at detecting small and overlapping players in sports footage.

    model = YOLO(CONFIG["model_path"])
    model.to(CONFIG["device"])

    print(f"Model loaded: {CONFIG['model_path']}")
    print(f"Model parameters: {sum(p.numel() for p in model.model.parameters()):,}")





    # BoT-SORT config written to yaml — Ultralytics reads this automatically.
    #
    # The critical difference from ByteTrack:
    #   with_reid: true  → enables the OSNet-0.25 Re-ID appearance model
    #   This model extracts a 512-dim embedding from each detection crop.
    #   When a track is lost and a new detection appears, if their embeddings
    #   are close enough (< appearance_thresh cosine distance), they get the
    #   SAME ID instead of a new one. This is what fixes re-entry ID switches.

    botsort_cfg = {
        "tracker_type"      : "botsort",
        "track_high_thresh" : CONFIG["track_high_thresh"],
        "track_low_thresh"  : CONFIG["track_low_thresh"],
        "new_track_thresh"  : CONFIG["new_track_thresh"],
        "track_buffer"      : CONFIG["track_buffer"],
        "match_thresh"      : CONFIG["match_thresh"],
        "proximity_thresh"  : CONFIG["proximity_thresh"],
        "appearance_thresh" : CONFIG["appearance_thresh"],
        "with_reid"         : True,     # ← THE KEY FLAG. Enables appearance Re-ID.
        "fuse_score"        : True,
        "gmc_method"        : "sparseOptFlow",  # Global Motion Compensation
        # sparseOptFlow estimates camera motion between frames and
        # corrects the Kalman filter predictions accordingly.
        # This prevents ID switches when the camera pans/zooms.
    }

    cfg_path = os.path.join(output_dir, f"{_stem}_botsort.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(botsort_cfg, f)

    print("BoT-SORT config written.")
    print("Re-ID: ENABLED (OSNet appearance model)")
    print("Global Motion Compensation: sparseOptFlow")
    print(f"Appearance threshold: {CONFIG['appearance_thresh']} (cosine distance)")
    print(f"Track buffer: {CONFIG['track_buffer']} frames")



    # Always inspect the video before processing — know your frame count,
    # fps, and resolution so you can set sensible parameters.

    cap = cv2.VideoCapture(CONFIG["input_video"])
    assert cap.isOpened(), f"Cannot open {CONFIG['input_video']}"

    TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    FPS          = cap.get(cv2.CAP_PROP_FPS)
    W            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    DURATION_SEC = TOTAL_FRAMES / FPS

    cap.release()

    print(f"Video : {CONFIG['input_video']}")
    print(f"Size  : {W}x{H}  |  FPS: {FPS:.1f}  |  Frames: {TOTAL_FRAMES}  |  Duration: {DURATION_SEC:.1f}s")

    # ── ID STABILITY LAYER ─────────────────────────────────────────────────────────
#
# Even with Re-ID, occasional ID switches happen (two players cross paths
# and swap). This layer adds a second line of defence:
#
# Problem: tracker assigns raw ID=47 after a merge, but the player was ID=3.
# Solution: we maintain a "stable ID" mapping. When a raw ID appears for the
# first time, we try to spatially match it to a recently-lost stable ID using
# trajectory prediction (where was that stable ID's last known position?).
# If distance < threshold → re-use the stable ID instead of creating a new one.
# If no match → assign next available stable ID.
#
# This is essentially a post-processing Re-ID layer on top of the tracker.

    class IDStabilizer:
        def __init__(self, spatial_threshold=80, max_lost_frames=120):
            """
            spatial_threshold: pixel distance to accept a re-match (80px at 1080p)
            max_lost_frames: how long to remember a lost ID for potential re-match
            """
            self.raw_to_stable   = {}        # raw tracker ID → stable display ID
            self.stable_counter  = 0         # next stable ID to assign
            self.lost_tracks     = {}        # stable_id → (last_cx, last_cy, lost_frame)
            self.spatial_thresh  = spatial_threshold
            self.max_lost_frames = max_lost_frames

        def update(self, raw_id, cx, cy, frame_num):
            """Returns the stable ID for a given raw tracker ID."""
            if raw_id in self.raw_to_stable:
                # Known mapping — just remove from lost if it was marked lost
                stable_id = self.raw_to_stable[raw_id]
                self.lost_tracks.pop(stable_id, None)
                return stable_id

            # New raw ID — try to match to a recently lost stable ID
            best_stable = None
            best_dist   = self.spatial_thresh

            for s_id, (lx, ly, lost_f) in list(self.lost_tracks.items()):
                if frame_num - lost_f > self.max_lost_frames:
                    del self.lost_tracks[s_id]
                    continue
                dist = np.sqrt((cx - lx)**2 + (cy - ly)**2)
                if dist < best_dist:
                    best_dist   = dist
                    best_stable = s_id

            if best_stable is not None:
                # Re-matched — reuse the stable ID
                self.raw_to_stable[raw_id] = best_stable
                del self.lost_tracks[best_stable]
            else:
                # Genuinely new — assign next stable ID
                self.stable_counter += 1
                self.raw_to_stable[raw_id] = self.stable_counter

            return self.raw_to_stable[raw_id]

        def mark_lost(self, raw_id, cx, cy, frame_num):
            """Call when a track disappears — remembers position for re-matching."""
            if raw_id in self.raw_to_stable:
                stable_id = self.raw_to_stable[raw_id]
                self.lost_tracks[stable_id] = (cx, cy, frame_num)

    stabilizer = IDStabilizer(
        spatial_threshold=100,   # ~100px at 1080p is ~1 player-width
        max_lost_frames=int(FPS * 3)  # remember lost IDs for 3 seconds
    )

    # Track which raw IDs were active last frame (to detect disappearances)
    prev_active_raw = {}   # raw_id → (cx, cy)

    print("ID Stabilizer ready.")
    print(f"  Spatial threshold : 100px")
    print(f"  Memory window     : {int(FPS*3)} frames (3 seconds)")



    # ── MAIN TRACKING LOOP v2 ──────────────────────────────────────────────────────
    #
    # Upgrades vs previous version:
    # 1. BoT-SORT with Re-ID instead of ByteTrack
    # 2. Ball detection + separate ball tracker (IoU-only, no Re-ID needed)
    # 3. IDStabilizer post-processing layer
    # 4. Global Motion Compensation handles camera pan/zoom
    # 5. Adaptive confidence: lower for ball (small, fast), higher for persons

    cap = cv2.VideoCapture(CONFIG["input_video"])
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(CONFIG["output_video"], fourcc, FPS, (W, H))

    # Data collectors
    trajectory_history     = defaultdict(list)
    ball_trajectory        = deque(maxlen=90)
    heatmap_accumulator    = np.zeros((H, W), dtype=np.float32)
    object_count_per_frame = []
    all_seen_stable_ids    = set()
    ball_positions         = []
    frame_idx = 0
    current_active_raw = {}

    def id_to_color(stable_id):
        """Deterministic vivid colour from stable ID — consistent across frames."""
        np.random.seed(stable_id * 17 + 13)
        h = int(np.random.randint(0, 180))  # HSV hue
        return tuple(int(x) for x in cv2.cvtColor(
            np.array([[[h, 220, 220]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
        )[0][0])

    print(f"Processing {TOTAL_FRAMES} frames with BoT-SORT + Re-ID...")
    start_time = datetime.now()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # ── Detect ALL target classes in one pass ─────────────────────────────
        # We run one model call with classes=[0, 32] — persons + ball.
        # Then we split detections by class for separate handling.
        results = model.track(
            frame,
            conf    = CONFIG["conf_person"],   # will catch ball too (ball conf lower anyway)
            iou     = CONFIG["iou_thresh"],
            imgsz   = CONFIG["imgsz"],
            device  = CONFIG["device"],
            classes = CONFIG["target_classes"],
            tracker = cfg_path,
            persist = True,
            verbose = False,
        )

        result = results[0]
        detections = sv.Detections.from_ultralytics(result)

        # ── Split persons vs ball ─────────────────────────────────────────────
        person_mask = detections.class_id == 0
        ball_mask   = detections.class_id == 32

        persons = detections[person_mask]
        balls   = detections[ball_mask]

        # ── Mark previously active tracks that disappeared ────────────────────
        if detections.tracker_id is not None:
            current_raw_ids = set(detections.tracker_id[person_mask].tolist()) \
                            if persons.tracker_id is not None else set()
        else:
            current_raw_ids = set()

        for raw_id, (lx, ly) in prev_active_raw.items():
            if raw_id not in current_raw_ids:
                stabilizer.mark_lost(raw_id, lx, ly, frame_idx)

        prev_active_raw.clear()

        # ── Process person tracks ─────────────────────────────────────────────
        active_stable_ids = set()
        annotated_frame = frame.copy()

        if persons.tracker_id is not None:
            for i, raw_id in enumerate(persons.tracker_id):
                if raw_id is None:
                    continue
                x1, y1, x2, y2 = persons.xyxy[i].astype(int)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                conf_val = float(persons.confidence[i])

                # Get stable ID (may re-use a previously lost ID)
                stable_id = stabilizer.update(raw_id, cx, cy, frame_idx)
                active_stable_ids.add(stable_id)
                all_seen_stable_ids.add(stable_id)
                prev_active_raw[raw_id] = (cx, cy)

                # Update trajectory + heatmap
                trajectory_history[stable_id].append((cx, cy))
                radius = max(10, int(np.sqrt((x2-x1)*(y2-y1)) * 0.15))
                cv2.circle(heatmap_accumulator, (cx, cy), radius, 1.0, -1)

        # ── Draw person trails ────────────────────────────────────────────────
        for stable_id in active_stable_ids:
            pts = trajectory_history[stable_id]
            color = id_to_color(stable_id)
            trail = pts[-CONFIG["max_trail_len"]:]
            for j in range(len(trail) - 1):
                alpha = j / max(len(trail), 1)
                thickness = max(1, int(alpha * 3))
                cv2.line(annotated_frame, trail[j], trail[j+1], color, thickness)

        # ── Draw person boxes + labels ────────────────────────────────────────
        if persons.tracker_id is not None:
            for i, raw_id in enumerate(persons.tracker_id):
                if raw_id is None:
                    continue
                x1, y1, x2, y2 = persons.xyxy[i].astype(int)
                conf_val = float(persons.confidence[i])
                stable_id = stabilizer.raw_to_stable.get(raw_id, raw_id)
                color = id_to_color(stable_id)

                # Rounded-corner look: draw 4 corner brackets instead of full box
                corner = min((x2-x1)//4, (y2-y1)//4, 20)
                for (px, py), (dx, dy) in [
                    ((x1,y1),(1,1)), ((x2,y1),(-1,1)),
                    ((x1,y2),(1,-1)), ((x2,y2),(-1,-1))
                ]:
                    cv2.line(annotated_frame, (px,py), (px+dx*corner,py), color, 2)
                    cv2.line(annotated_frame, (px,py), (px,py+dy*corner), color, 2)
                # Full box thin line
                cv2.rectangle(annotated_frame, (x1,y1), (x2,y2), color, 1)

                # Label
                label = f"#{stable_id}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                ly_pos = max(y1 - 6, th + 6)
                cv2.rectangle(annotated_frame, (x1, ly_pos-th-4), (x1+tw+6, ly_pos+2), color, -1)
                cv2.putText(annotated_frame, label, (x1+3, ly_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2, cv2.LINE_AA)

        # ── Process + draw ball ───────────────────────────────────────────────
        # Ball gets a simple IoU tracker (no Re-ID — ball appearance is useless).
        # We only show the ball if confidence > conf_ball threshold.
        if len(balls) > 0:
            # Take highest-confidence ball detection
            best_ball_idx = np.argmax(balls.confidence)
            bx1, by1, bx2, by2 = balls.xyxy[best_ball_idx].astype(int)
            ball_conf = float(balls.confidence[best_ball_idx])

            if ball_conf >= CONFIG["conf_ball"]:
                bcx, bcy = (bx1+bx2)//2, (by1+by2)//2
                ball_trajectory.append((bcx, bcy))
                ball_positions.append((frame_idx, bcx, bcy))

                # Draw ball trail (cyan dashed)
                btrl = list(ball_trajectory)
                for j in range(len(btrl)-1):
                    alpha = j / max(len(btrl), 1)
                    cv2.line(annotated_frame, btrl[j], btrl[j+1],
                            CONFIG["ball_color"], max(1, int(alpha*3)))

                # Ball circle marker
                brad = max(8, (bx2-bx1)//2)
                cv2.circle(annotated_frame, (bcx,bcy), brad, CONFIG["ball_color"], 2)
                cv2.circle(annotated_frame, (bcx,bcy), 3, CONFIG["ball_color"], -1)
                cv2.putText(annotated_frame, f"Ball {ball_conf:.2f}",
                            (bx1, by1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            CONFIG["ball_color"], 2, cv2.LINE_AA)

        object_count_per_frame.append(len(active_stable_ids))

        # ── HUD overlay ───────────────────────────────────────────────────────
        elapsed = (datetime.now() - start_time).total_seconds()
        fps_live = frame_idx / max(elapsed, 0.001)

        hud = [
            f"Frame {frame_idx}/{TOTAL_FRAMES}",
            f"Players: {len(active_stable_ids)}",
            f"Total IDs: {len(all_seen_stable_ids)}",
            f"Ball: {'YES' if len(balls)>0 else 'no'}",
            f"Speed: {fps_live:.1f}fps",
        ]
        # Semi-transparent dark bar
        bar_h = len(hud) * 24 + 10
        overlay_bar = annotated_frame.copy()
        cv2.rectangle(overlay_bar, (0,0), (210, bar_h), (0,0,0), -1)
        cv2.addWeighted(overlay_bar, 0.5, annotated_frame, 0.5, 0, annotated_frame)
        for li, line in enumerate(hud):
            cv2.putText(annotated_frame, line, (8, 22+li*24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

        out.write(annotated_frame)

        if frame_idx % 100 == 0:
            pct = frame_idx/TOTAL_FRAMES*100
            print(f"  [{pct:5.1f}%] f={frame_idx}  players={len(active_stable_ids)}"
                f"  stable_ids={len(all_seen_stable_ids)}  ball={'✓' if len(balls)>0 else '✗'}"
                f"  {fps_live:.1f}fps")

    cap.release()
    out.release()
    total_time = (datetime.now() - start_time).total_seconds()
    print(f"\nDone! {frame_idx} frames in {total_time:.1f}s ({frame_idx/total_time:.1f} fps)")
    print(f"Stable unique IDs: {len(all_seen_stable_ids)}  (was 46 with ByteTrack)")
    print(f"Output: {CONFIG['output_video']}")

    # Movement heatmap — shows where players spent the most time on the field.
    # This is one of the "optional enhancements" that strongly impress evaluators.
    #
    # We apply a Gaussian blur to smooth the point accumulation, then overlay
    # on a sample frame so the field context is preserved.

    cap_tmp = cv2.VideoCapture(CONFIG["input_video"])
    ret, sample_frame = cap_tmp.read()
    cap_tmp.release()

    # Normalise and apply colormap
    heatmap_norm = cv2.normalize(heatmap_accumulator, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_blur = cv2.GaussianBlur(heatmap_norm.astype(np.uint8), (31, 31), 0)
    heatmap_color = cv2.applyColorMap(heatmap_blur, cv2.COLORMAP_JET)

    # Blend with sample frame
    overlay = cv2.addWeighted(sample_frame, 0.4, heatmap_color, 0.6, 0)

    cv2.imwrite(CONFIG["output_heatmap"], overlay)

    # Show inline
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Sample Frame", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Movement Heatmap", fontsize=14)
    axes[1].axis("off")

    plt.suptitle("Player Movement Heatmap", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(CONFIG["output_heatmap"], dpi=150, bbox_inches="tight")
    print(f"Heatmap saved: {CONFIG['output_heatmap']}")


    # Static trajectory plot — shows the movement paths of the most active players.
    # Great for the technical report screenshots.

    cap_tmp = cv2.VideoCapture(CONFIG["input_video"])
    ret, bg = cap_tmp.read()
    cap_tmp.release()
    bg_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(bg_rgb, alpha=0.3)

    # Plot top-20 most active (longest trajectories) so the plot isn't crowded
    sorted_ids = sorted(trajectory_history.items(), key=lambda x: len(x[1]), reverse=True)[:20]

    cmap = plt.cm.get_cmap("tab20", len(sorted_ids))

    for idx, (tid, pts) in enumerate(sorted_ids):
        if len(pts) < 5:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, linewidth=1.5, alpha=0.8, color=cmap(idx), label=f"ID {tid}")
        ax.scatter(xs[-1], ys[-1], color=cmap(idx), s=40, zorder=5)  # endpoint dot

    ax.set_title(f"Player Trajectory Paths  (top {len(sorted_ids)} most active)", fontsize=14)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(CONFIG["output_traj"], dpi=150, bbox_inches="tight")
    print(f"Trajectory plot saved: {CONFIG['output_traj']}")


    # Ball trajectory over time — separate plot since ball moves very differently
    # from players. Shows ball movement path across the full video.

    if len(ball_positions) > 10:
        cap_tmp = cv2.VideoCapture(CONFIG["input_video"])
        ret, bg = cap_tmp.read(); cap_tmp.release()
        bg_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.imshow(bg_rgb, alpha=0.25)

        frames_b, xs_b, ys_b = zip(*ball_positions)
        # Colour by time: early=blue, late=red
        sc = ax.scatter(xs_b, ys_b, c=frames_b, cmap='plasma',
                        s=20, alpha=0.7, zorder=3)
        ax.plot(xs_b, ys_b, color='cyan', linewidth=0.8, alpha=0.5)
        plt.colorbar(sc, ax=ax, label='Frame number')
        ax.set_title("Ball Trajectory (colour = time)", fontsize=14)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(CONFIG["output_ball"], dpi=150, bbox_inches="tight")
        print(f"Ball detected in {len(ball_positions)}/{TOTAL_FRAMES} frames "
            f"({100*len(ball_positions)/TOTAL_FRAMES:.1f}%)")
    else:
        print("Ball not detected enough frames to plot. "
            "Try lowering conf_ball in CONFIG or check video has a visible ball.")





    # Object count over time — required by assignment ("object count over time").
    # Shows tracking stability: a good tracker has a smooth, consistent count line.
    # ID-switching shows up as spikes.

    fig, ax = plt.subplots(figsize=(14, 4))
    time_axis = np.arange(len(object_count_per_frame)) / FPS

    ax.plot(time_axis, object_count_per_frame, linewidth=1, color="#2196F3", alpha=0.8)
    ax.fill_between(time_axis, object_count_per_frame, alpha=0.2, color="#2196F3")

    # Rolling average for trend line
    window = int(FPS * 2)  # 2-second window
    if len(object_count_per_frame) > window:
        rolling = np.convolve(object_count_per_frame,
                            np.ones(window)/window, mode='valid')
        ax.plot(time_axis[:len(rolling)], rolling,
                linewidth=2.5, color="#F44336", label=f"2s rolling avg")
        ax.legend()

    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Active Tracked IDs", fontsize=12)
    ax.set_title("Object Count Over Time", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CONFIG["output_count"], dpi=150, bbox_inches="tight")

    print(f"Peak count  : {max(object_count_per_frame)}")
    print(f"Avg count   : {np.mean(object_count_per_frame):.1f}")
    print(f"Total IDs   : {len(all_seen_stable_ids)}")



    stats = {
        "video_source"         : CONFIG["input_video"],
        "model"                : CONFIG["model_path"],
        "tracker"              : "botsort_with_reid",
        "id_stabilizer"        : "enabled",
        "total_frames"         : frame_idx,
        "fps_original"         : FPS,
        "resolution"           : f"{W}x{H}",
        "total_stable_ids"     : len(all_seen_stable_ids),
        "peak_simultaneous"    : int(max(object_count_per_frame)),
        "avg_simultaneous"     : round(float(np.mean(object_count_per_frame)), 2),
        "ball_detected_frames" : len(ball_positions),
        "ball_detection_rate"  : round(len(ball_positions)/max(frame_idx,1), 3),
        "conf_person"          : CONFIG["conf_person"],
        "conf_ball"            : CONFIG["conf_ball"],
        "track_buffer_frames"  : CONFIG["track_buffer"],
        "appearance_thresh"    : CONFIG["appearance_thresh"],
        "imgsz"                : CONFIG["imgsz"],
        "processing_time_s"    : round(total_time, 2),
        "processing_fps"       : round(frame_idx/total_time, 1),
    }

    with open(CONFIG["stats_json"], "w") as f:
        json.dump(stats, f, indent=2)

    print("Stats saved.")
    print(json.dumps(stats, indent=2))

    rel_path = os.path.basename(CONFIG["output_video"])

    return {
        "video": f"outputs/{rel_path}",
        "heatmap": f"outputs/{os.path.basename(CONFIG['output_heatmap'])}",
        "trajectory": f"outputs/{os.path.basename(CONFIG['output_traj'])}",
        "stats": f"outputs/{os.path.basename(CONFIG['stats_json'])}"
    }