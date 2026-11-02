from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor


def main(request_path: str) -> None:
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    output_dir = Path(request["output_dir"])
    input_frame_dir = output_dir / "input_frames"
    frame_dir = output_dir / "frames"
    mask_dir = output_dir / "masks"
    overlay_dir = output_dir / "overlays"
    for folder in (input_frame_dir, frame_dir, mask_dir, overlay_dir):
        folder.mkdir(parents=True, exist_ok=True)

    fps = max(1, min(8, int(request.get("fps", 4))))
    window_seconds = max(1, min(30, int(request.get("window_seconds", 30))))
    start_s = max(0.0, float(request["seed_timestamp_s"]))
    segment_end = request.get("segment_end_s")
    duration = float(window_seconds)
    if segment_end is not None:
        duration = min(duration, max(0.0, float(segment_end) - start_s))
    if duration <= 0:
        raise ValueError("selected frame is at the end of the video")

    pattern = str(input_frame_dir / "%05d.jpg")
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start_s:.4f}", "-i", request["video"], "-t", f"{duration:.4f}",
        "-vf", f"fps={fps}", "-q:v", "2", "-start_number", "0", pattern,
    ], check=True, capture_output=True, text=True)
    frame_paths = sorted(input_frame_dir.glob("*.jpg"))
    if not frame_paths:
        raise RuntimeError("ffmpeg did not extract any video frames")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        request["checkpoint"],
        device=device,
    )
    state = model.init_state(
        video_path=str(input_frame_dir),
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
    )
    points = request["points"]
    with Image.open(frame_paths[0]) as first:
        width, height = first.size
    for point in points:
        if not (0 <= float(point["x"]) < width and 0 <= float(point["y"]) < height):
            raise ValueError("point prompt is outside the selected video frame")

    with torch.inference_mode():
        model.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=1,
            points=np.asarray([[p["x"] for p in points], [p["y"] for p in points]], dtype=np.float32).T,
            labels=np.asarray([p["label"] for p in points], dtype=np.int32),
            normalize_coords=True,
        )
        prefix_frames = request.get("prefix_frames") or []
        prefix_count = len(prefix_frames)
        frame_rows = []
        parent_dir = Path(request["parent_run_dir"]) if request.get("parent_run_dir") else None
        for frame in prefix_frames:
            index = int(frame["index"])
            for folder, suffix in (("frames", "jpg"), ("masks", "png"), ("overlays", "png")):
                source_path = parent_dir / folder / f"{index:05d}.{suffix}"
                if not source_path.is_file():
                    raise FileNotFoundError(f"parent track is missing {folder}/{index:05d}.{suffix}")
                shutil.copy2(source_path, output_dir / folder / source_path.name)
            frame_rows.append(dict(frame))

        for frame_index, object_ids, logits in model.propagate_in_video(
            state, start_frame_idx=0, max_frame_num_to_track=len(frame_paths)
        ):
            output_index = prefix_count + int(frame_index)
            shutil.copy2(frame_paths[frame_index], frame_dir / f"{output_index:05d}.jpg")
            if 1 not in object_ids:
                mask = np.zeros((height, width), dtype=np.uint8)
                frame_status = "lost"
            else:
                object_index = list(object_ids).index(1)
                mask = (logits[object_index] > 0.0).squeeze().to("cpu").numpy().astype(np.uint8) * 255
                frame_status = "tracked" if int(mask.sum()) else "empty"
            Image.fromarray(mask, mode="L").save(mask_dir / f"{output_index:05d}.png")
            rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
            rgba[..., :3] = (49, 130, 206)
            rgba[..., 3] = (mask.astype(np.float32) * 0.48).astype(np.uint8)
            Image.fromarray(rgba, mode="RGBA").save(overlay_dir / f"{output_index:05d}.png")
            frame_rows.append({
                "index": output_index,
                "timestamp_s": round(start_s + frame_index / fps, 3),
                "foreground_pixels": int(mask.sum() // 255),
                "status": frame_status,
            })

    frame_rows.sort(key=lambda row: int(row["index"]))
    shutil.rmtree(input_frame_dir, ignore_errors=True)
    previous_area = None
    stable_areas = []
    previous_visibility = None
    frame_area_limit = max(1, width * height)
    for row in frame_rows:
        area = int(row.get("foreground_pixels") or 0)
        reason = None
        baseline = float(np.median(stable_areas[-5:])) if stable_areas else 0.0
        if area == 0 or row.get("status") in {"lost", "empty"}:
            visibility = "lost"
            visibility_reason = "mask_missing"
        elif previous_visibility == "lost":
            visibility = "reappeared"
            visibility_reason = "mask_returned"
            stable_areas.append(area)
        elif baseline and area < baseline / 6:
            visibility = "suspected_occluded"
            visibility_reason = "area_drop_vs_recent_visible"
        else:
            visibility = "visible"
            visibility_reason = "mask_present"
            stable_areas.append(area)
        row["visibility_status"] = visibility
        row["visibility_reason"] = visibility_reason

        if area == 0 or row.get("status") in {"lost", "empty"}:
            reason = "empty_or_lost"
        elif visibility == "suspected_occluded":
            reason = "suspected_occlusion"
        elif area < 16:
            reason = "tiny_mask"
        elif area > int(frame_area_limit * 0.9):
            reason = "near_full_frame"
        elif previous_area and max(area, previous_area) / max(1, min(area, previous_area)) > 6:
            reason = "area_jump"
        row["review_suggested"] = reason is not None
        row["review_reason"] = reason

        previous_visibility = visibility
        if area and visibility == "visible":
            previous_area = area

    result_start = float(request.get("original_start_timestamp_s", start_s))
    requested_window_s = int(request.get("requested_window_s", window_seconds))
    lost_frames = sum(
        row.get("status") in {"lost", "empty"}
        or int(row.get("foreground_pixels") or 0) == 0
        for row in frame_rows
    )

    result = {
        "device": device,
        "fps": fps,
        "start_timestamp_s": round(result_start, 3),
        "end_timestamp_s": frame_rows[-1]["timestamp_s"] if frame_rows else round(start_s, 3),
        "duration_s": round(len(frame_rows) / fps, 3),
        "requested_window_s": requested_window_s,
        "frame_count": len(frame_rows),
        "lost_frames": lost_frames,
    "suspected_occlusion_frames": sum(row.get("visibility_status") == "suspected_occluded" for row in frame_rows),
    "visibility_counts": {
        status: sum(row.get("visibility_status") == status for row in frame_rows)
        for status in ("visible", "suspected_occluded", "lost", "reappeared")
    },
        "review_suggested_frames": sum(bool(row.get("review_suggested")) for row in frame_rows),
        "frames": frame_rows,
    }
    metadata = {
        "id": output_dir.name,
        "job_id": request.get("job_id"),
        "status": "completed",
        "model": "SAM2.1 Hiera Small Video Predictor",
        "checkpoint": Path(request["checkpoint"]).name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": request.get("source"),
        "points": request.get("parent_points", request.get("points", [])),
        "parent_run_id": request.get("parent_run_id"),
        "corrections": request.get("corrections", []),
        "correction": request.get("correction"),
        **result,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: sam2_track_video.py request.json")
    main(sys.argv[1])
