from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from florence_runtime import load_florence, predict_boxes
from sam2_segment_image import SegmentRuntime


def write_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def update_progress(path: Path, **values) -> None:
    current = {}
    if path.is_file():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    current.update(values)
    write_json(path, current)


def extract_episode_frames(episode: dict, interval_s: float, output_dir: Path) -> list[Path]:
    from subprocess import run

    frame_dir = output_dir / f"episode_{int(episode['episode']):06d}" / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old in frame_dir.glob("frame_*.jpg"):
        old.unlink()
    start = float(episode["sample_start_s"])
    end = float(episode["sample_end_s"])
    if end <= start:
        return []
    pattern = str(frame_dir / "frame_%05d.jpg")
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.4f}", "-i", episode["video"],
        "-t", f"{end - start:.4f}",
        "-vf", f"fps={1.0 / interval_s:.8f}",
        "-q:v", "2", "-start_number", "0", pattern,
    ], check=True, capture_output=True, text=True)
    return sorted(frame_dir.glob("frame_*.jpg"))


def color_for(index: int) -> tuple[int, int, int]:
    colors = ((49, 130, 206), (225, 139, 37), (138, 92, 246), (22, 163, 106), (224, 72, 77))
    return colors[index % len(colors)]


def main(request_path: str) -> None:
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    output_dir = Path(request["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"
    results_path = output_dir / "results.json"
    interval_s = float(request["interval_s"])
    threshold = float(request["threshold"])
    started = time.monotonic()
    frame_plans = []
    results = {
        "id": request["batch_id"],
        "job_id": int(os.environ.get("EF_JOB_ID") or request.get("job_id") or 0),
        "dataset_version_id": int(request["dataset_version_id"]),
        "camera": request["camera"],
        "prompt": request["prompt"],
        "threshold": threshold,
        "detector": request.get("detector", "florence-2-large"),
        "text_threshold": request.get("text_threshold", 0.25),
        "interval_s": interval_s,
        "episodes": [int(ep["episode"]) for ep in request["episodes"]],
        "total_frames": 0,
        "completed_frames": 0,
        "filtered_low_confidence": 0,
        "detections": [],
        "frames": [],
        "created_at": request.get("created_at"),
    }
    try:
        update_progress(progress_path, status="extracting", total_frames=int(request.get("estimated_frames") or 0), completed_frames=0)
        for episode in request["episodes"]:
            paths = extract_episode_frames(episode, interval_s, output_dir)
            frame_plans.extend((episode, index, path) for index, path in enumerate(paths))
        if not frame_plans:
            raise RuntimeError("所选范围没有可抽取的视频帧")
        results["total_frames"] = len(frame_plans)
        write_json(results_path, results)
        update_progress(progress_path, status="loading_models", total_frames=len(frame_plans), completed_frames=0)

        use_dino = request.get("detector") == "grounding-dino-tiny"
        runtime = SegmentRuntime() if use_dino else load_florence(request["model_dir"])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam_model = build_sam2(
            "configs/sam2.1/sam2.1_hiera_s.yaml", request["checkpoint"], device=device
        )
        predictor = SAM2ImagePredictor(sam_model)
        results["device"] = device if use_dino else runtime["device"]
        results["sam2_device"] = device
        update_progress(progress_path, status="running", total_frames=len(frame_plans), completed_frames=0)
        total_florence_s = 0.0
        total_sam2_s = 0.0

        for done, (episode, frame_index, frame_path) in enumerate(frame_plans, start=1):
            image = Image.open(frame_path).convert("RGB")
            ts = round(float(episode["sample_start_s"]) + frame_index * interval_s, 3)
            frame_started = time.monotonic()
            if use_dino:
                all_boxes = runtime.ground_dino({
                    "model_dir": request["model_dir"], "image": str(frame_path),
                    "prompt": request["prompt"], "threshold": threshold,
                    "text_threshold": request.get("text_threshold", 0.25),
                })["boxes"]
            else:
                all_boxes = predict_boxes(image, request["prompt"], runtime)
            total_florence_s += time.monotonic() - frame_started
            kept = [item for item in all_boxes if item.get("score") is None or item["score"] >= threshold]
            filtered = len(all_boxes) - len(kept)
            results["filtered_low_confidence"] += filtered
            episode_id = int(episode["episode"])
            episode_dir = output_dir / f"episode_{episode_id:06d}"
            overlay = np.asarray(image).copy()
            frame_row = {
                "episode": episode_id,
                "camera": request["camera"],
                "frame_index": frame_index,
                "timestamp_s": ts,
                "filename": frame_path.name,
                "relative_frame_path": frame_path.relative_to(output_dir).as_posix(),
                "filtered_low_confidence": filtered,
                "detections": [],
            }
            if kept:
                predictor.set_image(np.asarray(image))
                for detection_index, detection in enumerate(kept):
                    b = detection["box"]
                    box = np.asarray([b["x1"], b["y1"], b["x2"], b["y2"]], dtype=np.float32)
                    sam_started = time.monotonic()
                    with torch.inference_mode():
                        masks, scores, _ = predictor.predict(box=box, multimask_output=True)
                    total_sam2_s += time.monotonic() - sam_started
                    best = int(np.argmax(scores))
                    mask = (masks[best].astype(np.uint8) * 255)
                    mask_rel = Path(f"episode_{episode_id:06d}") / "masks" / f"frame_{frame_index:05d}_det_{detection_index:02d}.png"
                    overlay_rel = Path(f"episode_{episode_id:06d}") / "overlays" / f"frame_{frame_index:05d}.jpg"
                    mask_path = output_dir / mask_rel
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(mask, mode="L").save(mask_path)
                    color = np.asarray(color_for(detection_index), dtype=np.float32)
                    active = mask > 0
                    overlay[active] = (overlay[active].astype(np.float32) * 0.52 + color * 0.48).astype(np.uint8)
                    frame_row["detections"].append({
                        "detection_index": detection_index,
                        "label": detection["label"],
                        "score": detection.get("score"),
                        "box": b,
                        "mask_predicted_iou": float(scores[best]),
                        "foreground_pixels": int(mask.sum() // 255),
                        "relative_mask_path": mask_rel.as_posix(),
                    })
                    results["detections"].append({
                        "episode": episode_id,
                        "camera": request["camera"],
                        "frame_index": frame_index,
                        "detection_index": detection_index,
                        "timestamp_s": ts,
                        "label": detection["label"],
                        "score": detection.get("score"),
                        "box": b,
                        "mask_predicted_iou": float(scores[best]),
                        "foreground_pixels": int(mask.sum() // 255),
                        "relative_frame_path": frame_row["relative_frame_path"],
                        "relative_mask_path": mask_rel.as_posix(),
                    })
            if kept:
                overlay_path = output_dir / overlay_rel
                overlay_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(overlay, mode="RGB").save(overlay_path, quality=88)
                frame_row["relative_overlay_path"] = overlay_rel.as_posix()
            results["frames"].append(frame_row)
            results["completed_frames"] = done
            results["florence_seconds"] = round(total_florence_s, 3)
            results["sam2_seconds"] = round(total_sam2_s, 3)
            write_json(results_path, results)
            update_progress(
                progress_path, status="running", total_frames=len(frame_plans),
                completed_frames=done, current_episode=episode_id,
                current_timestamp_s=ts, current_detections=len(kept),
            )

        results.update({
            "status": "completed",
            "finished_at": time.time(),
            "elapsed_s": round(time.monotonic() - started, 3),
            "florence_seconds": round(total_florence_s, 3),
            "sam2_seconds": round(total_sam2_s, 3),
        })
        write_json(results_path, results)
        update_progress(progress_path, status="completed", total_frames=len(frame_plans), completed_frames=len(frame_plans))
        print(json.dumps({
            "batch_id": request["batch_id"], "total_frames": len(frame_plans),
            "detections": len(results["detections"]),
            "filtered_low_confidence": results["filtered_low_confidence"],
            "elapsed_s": results["elapsed_s"], "device": results["device"],
        }, ensure_ascii=False))
    except Exception as exc:
        update_progress(progress_path, status="failed", error=str(exc)[:1200])
        results.update({"status": "failed", "error": str(exc)[:1200]})
        write_json(results_path, results)
        raise


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: florence_ground_batch.py request.json")
    main(sys.argv[1])
