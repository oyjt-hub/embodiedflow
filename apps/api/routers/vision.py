from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import uuid
import zipfile
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pydantic import ValidationError
from sqlalchemy import select

from core.config import settings
from persistence.db import session_scope
from persistence.models import Annotation, AnnotationRun, DatasetVersion, Evaluation, Job, PipelineTask, utcnow
from services import jobs as jobsvc


router = APIRouter(prefix="/workbench/api/vision", tags=["vision"])
TORCH_PY = Path(os.environ.get("EMBODIEDFLOW_VISION_PYTHON",
                              str(settings.root / ".vision-venv" / "bin" / "python")))
SCRIPT = settings.root / "scripts" / "sam2_segment_image.py"
TRACK_SCRIPT = settings.root / "scripts" / "sam2_track_video.py"
CHECKPOINT = settings.data_root / "models" / "vision" / "sam2.1_hiera_small.pt"
FLORENCE_DIR = settings.data_root / "models" / "vision" / "florence-2-large"
DINO_DIR = settings.data_root / "models" / "vision" / "grounding-dino-tiny"
GROUND_SCRIPT = settings.root / "scripts" / "florence_ground.py"
GROUND_BATCH_SCRIPT = settings.root / "scripts" / "florence_ground_batch.py"
VISION_DIR = settings.data_root / "artifacts" / "vision"
FLORA_ROOT = Path.home() / "workspace" / "flora"
FLORA_ANNOTATE = FLORA_ROOT / "annotate.sh"
FLORA_SETUP = FLORA_ROOT / "status" / "setup_job.json"


class Point(BaseModel):
    x: float
    y: float
    label: int


class VisionSource(BaseModel):
    dataset_version_id: int
    episode: int
    camera: str
    timestamp_s: float


class Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class SegmentIn(BaseModel):
    image_base64: str
    filename: str = "image.png"
    points: list[Point] = Field(default_factory=list)
    box: Box | None = None
    source: VisionSource | None = None
    parent_run_id: str | None = None
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    label: str = Field(default="object", min_length=1, max_length=128)


class GroundIn(BaseModel):
    image_base64: str
    filename: str = "image.png"
    prompt: str = Field(min_length=1, max_length=500)
    threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    source: VisionSource | None = None
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    detector: str = Field(default="florence-2-large", pattern="^(florence-2-large|grounding-dino-tiny)$")
    text_threshold: float = Field(default=0.25, ge=0, le=1)


class GroundBatchIn(BaseModel):
    dataset_version_id: int
    episodes: list[int] = Field(min_length=1, max_length=128)
    camera: str
    prompt: str = Field(min_length=1, max_length=500)
    threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    detector: str = Field(default="florence-2-large", pattern="^(florence-2-large|grounding-dino-tiny)$")
    text_threshold: float = Field(default=0.25, ge=0, le=1)
    interval_s: float = Field(default=2.0, ge=0.25, le=60.0)
    start_s: float = Field(default=0.0, ge=0.0, le=86400.0)
    end_s: float | None = Field(default=None, gt=0.0, le=86400.0)


class AnnotationRunIn(BaseModel):
    dataset_version_id: int
    name: str = Field(default="bbox-gt-v1", min_length=1, max_length=32)


class GroundTruthBoxIn(BaseModel):
    label: str = Field(min_length=1, max_length=128)
    box: Box


class GroundTruthFrameIn(BaseModel):
    batch_id: str
    episode: int = Field(ge=0)
    camera: str = Field(min_length=1, max_length=128)
    frame_index: int = Field(ge=0)
    timestamp_s: float = Field(ge=0.0)
    width: int = Field(gt=0, le=16384)
    height: int = Field(gt=0, le=16384)
    boxes: list[GroundTruthBoxIn] = Field(default_factory=list, max_length=128)
    reviewer: str = Field(default="", max_length=64)


class GroundBenchmarkIn(BaseModel):
    annotation_run_id: int
    iou_threshold: float = Field(default=0.5, ge=0.1, le=0.99)


class TrackIn(BaseModel):
    source: VisionSource
    points: list[Point]


class TrackCorrectionIn(BaseModel):
    source: VisionSource
    points: list[Point]


class TrackBatchIn(BaseModel):
    source: VisionSource
    episodes: list[int] = Field(min_length=1, max_length=128)
    points: list[Point]


class FloraJobIn(BaseModel):
    dataset_version_id: int
    episode: int
    all_episodes: bool = False
    camera: str
    profile: str = "yam"
    instruction: str = ""
    episodes: list[int] | None = Field(default=None, min_length=1, max_length=10000)


class FloraReviewSegment(BaseModel):
    start_s: float
    end_s: float
    caption: str = Field(default="", max_length=2000)


class FloraReviewEpisode(BaseModel):
    episode: int
    segments: list[FloraReviewSegment] = Field(default_factory=list)
    mask_run_ids: list[str] = Field(default_factory=list)


class FloraReviewIn(BaseModel):
    annotations: list[FloraReviewEpisode]
    notes: str = Field(default="", max_length=4000)


class VisionOperationIn(BaseModel):
    operation: str = Field(pattern="^(segment_image|ground_image|ground_and_segment|track_video|batch_annotate)$")
    arguments: dict


@router.post("/execute")
def execute_vision_operation(payload: VisionOperationIn):
    try:
        if payload.operation == "segment_image":
            result = segment(SegmentIn.model_validate(payload.arguments))
        elif payload.operation == "ground_image":
            result = ground(GroundIn.model_validate(payload.arguments))
        elif payload.operation == "track_video":
            result = track_video(TrackIn.model_validate(payload.arguments))
        elif payload.operation == "batch_annotate":
            result = create_ground_batch(GroundBatchIn.model_validate(payload.arguments))
        else:
            arguments = GroundIn.model_validate(payload.arguments)
            detections = ground(arguments)
            instances = []
            for detection in detections.get("boxes", []):
                mask = segment(SegmentIn(
                    image_base64=arguments.image_base64, filename=arguments.filename,
                    source=arguments.source, session_id=arguments.session_id,
                    label=str(detection["label"])[:128] or "object",
                    box=Box.model_validate(detection["box"])))
                instances.append({"label": detection["label"], "detector_score": detection.get("score"),
                                  "box": detection["box"], "mask": mask})
            result = {"grounding": detections, "instances": instances, "count": len(instances)}
        return {"operation": payload.operation, "adapter_version": "1.0.0", "result": result}
    except ValidationError as exc:
        raise HTTPException(422, exc.errors(include_url=False)) from exc


def _dino_ready():
    try:
        state = json.loads((DINO_DIR / "download_status.json").read_text(encoding="utf-8"))
        return state.get("state") == "complete" and all((DINO_DIR / name).is_file()
            for name in ("config.json", "preprocessor_config.json", "model.safetensors"))
    except (OSError, json.JSONDecodeError):
        return False


@router.get("/health")
def health():
    ready = TORCH_PY.is_file() and SCRIPT.is_file() and CHECKPOINT.is_file()
    grounding_ready = TORCH_PY.is_file() and GROUND_SCRIPT.is_file() and all(
        (FLORENCE_DIR / name).is_file() for name in (
            "config.json", "preprocessor_config.json", "processing_florence2.py",
            "configuration_florence2.py", "modeling_florence2.py", "model.safetensors",
        )
    )
    return {
        "ready": ready,
        "video_tracking_ready": ready and TRACK_SCRIPT.is_file() and shutil.which("ffmpeg") is not None,
        "model": "SAM2.1 Hiera Small",
        "grounding_ready": grounding_ready,
        "ground_batch_ready": grounding_ready and GROUND_BATCH_SCRIPT.is_file()
        and CHECKPOINT.is_file() and shutil.which("ffmpeg") is not None,
        "grounding_model": "Florence-2-large",
        "dino_ready": _dino_ready(),
    }


@router.get("/capabilities")
def vision_capabilities():
    current = health()
    def asset(path):
        path = Path(path)
        stat = path.stat() if path.is_file() else None
        return {"path": str(path), "exists": stat is not None,
                "bytes": stat.st_size if stat else None,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat() if stat else None}
    return {
        "schema": "embodiedflow.vision-capabilities", "version": "1.0.0",
        "execute_url": "/workbench/api/vision/execute",
        "operations": ["segment_image", "ground_image", "ground_and_segment", "track_video", "batch_annotate"],
        "models": [
            {"id": "sam2.1-hiera-small", "name": "SAM2.1 Hiera Small", "ready": current["ready"],
             "tasks": ["point_segmentation", "box_segmentation", "video_tracking", "keyframe_correction"],
             "checkpoint": asset(CHECKPOINT), "python": str(TORCH_PY),
             "interactive_runtime": "leased_persistent_process", "batch_runtime": "gpu_worker",
             "exports": ["mask_png", "tracking_json", "zip"]},
            {"id": "florence-2-large", "name": "Florence-2-large", "ready": current["grounding_ready"],
             "tasks": ["text_detection", "text_detection_sam2_batch"],
             "checkpoint": asset(FLORENCE_DIR / "model.safetensors"), "python": str(TORCH_PY),
             "interactive_runtime": "leased_persistent_process", "batch_runtime": "gpu_worker",
             "parameters": {"threshold": {"min": 0, "max": 1, "default": 0.2}},
             "exports": ["boxes_json", "mask_png", "zip"]},
            {"id": "flora", "name": "NVIDIA FLORA", "ready": flora_status()["ready"],
             "tasks": ["temporal_action_segmentation", "robot_action_captioning"],
             "profiles": ["yam", "dex", "tianji", "human"], "runtime": "gpu_worker",
             "review_revisions": True, "exports": ["json", "csv", "srt", "zip", "lerobot_v2.1"]},
            {"id": "grounding-dino-tiny", "name": "Grounding DINO Tiny", "ready": _dino_ready(),
             "tasks": ["text_detection", "ground_and_segment"], "checkpoint": asset(DINO_DIR / "model.safetensors"),
             "parameters": {"threshold": {"min": 0, "max": 1, "default": 0.25},
                            "text_threshold": {"min": 0, "max": 1, "default": 0.25}},
             "interactive_runtime": "leased_persistent_process", "state": "ready" if _dino_ready() else "assets_pending"},
        ],
        "persistent_interactive_runtime": {"sam2": True, "florence": True},
        "review_api": "/workbench/api/vision/flora/jobs/{job_id}/review",
    }


@router.get("/runtime")
def get_vision_runtime():
    """运行时常驻状态。若标记为 active 但进程已死，自动清理（异常恢复）。"""
    import os as _os

    from services.vision_runtime import runtime

    snap = runtime.snapshot()
    pid = snap.get("pid")
    if snap.get("active") and pid:
        alive = True
        try:
            _os.kill(int(pid), 0)
        except (ProcessLookupError, PermissionError):
            alive = False
        except (TypeError, ValueError):
            alive = True
        if not alive:
            runtime.close()
            snap = runtime.snapshot()
            snap["recovered"] = "stale_pid_cleared"
    return snap


@router.post("/runtime/release")
def release_vision_runtime(session_id: str | None = None):
    from services.vision_runtime import runtime
    if session_id:
        return runtime.release(session_id)
    runtime.close()
    return runtime.snapshot()


def _run_dir(run_id: str) -> Path:
    try:
        safe_id = str(uuid.UUID(run_id))
    except ValueError as exc:
        raise HTTPException(404, "mask not found") from exc
    return VISION_DIR / safe_id


@router.post("/segment", status_code=201)
def segment(payload: SegmentIn):
    if not payload.box and (not payload.points or not any(p.label == 1 for p in payload.points)):
        raise HTTPException(400, "add a box or at least one positive point")
    if not TORCH_PY.is_file() or not SCRIPT.is_file() or not CHECKPOINT.is_file():
        raise HTTPException(503, "SAM2 runtime or checkpoint is not ready")
    raw = payload.image_base64.split(",", 1)[-1]
    try:
        image_bytes = base64.b64decode(raw, validate=True)
    except Exception as exc:
        raise HTTPException(400, "invalid image data") from exc

    if payload.parent_run_id:
        parent_dir = _run_dir(payload.parent_run_id)
        parent_image = parent_dir / "image.png"
        if not parent_image.is_file() or hashlib.sha256(parent_image.read_bytes()).digest() != hashlib.sha256(image_bytes).digest():
            raise HTTPException(400, "parent Mask must use the same source image")
    run_id = str(uuid.uuid4())
    out_dir = VISION_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / "image.png"
    mask_path = out_dir / "mask.png"
    request_path = out_dir / "request.json"
    image_path.write_bytes(image_bytes)
    mask_input = None
    if payload.parent_run_id and not payload.box:
        prior_logits = _run_dir(payload.parent_run_id) / "logits.npy"
        if prior_logits.is_file():
            mask_input = str(prior_logits)
    request_path.write_text(json.dumps({
        "image": str(image_path),
        "mask": str(mask_path),
        "checkpoint": str(CHECKPOINT),
        "points": [p.model_dump() for p in payload.points],
        "mask_input": mask_input,
        "box": payload.box.model_dump() if payload.box else None,
    }), encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    vendor = str(settings.root / "vendor" / "sam2")
    env["PYTHONPATH"] = vendor + os.pathsep + env.get("PYTHONPATH", "")
    try:
        from services.vision_runtime import runtime
        result = runtime.execute(json.loads(request_path.read_text(encoding="utf-8")),
                                 TORCH_PY, SCRIPT, vendor, settings.data_root,
                                 owner=payload.session_id or "api")
    except Exception as exc:
        raise HTTPException(503, str(exc)) from exc

    metadata = {
        "id": run_id,
        "filename": Path(payload.filename).name,
        "parent_run_id": payload.parent_run_id,
        "mask_prompt_run_id": payload.parent_run_id if mask_input else None,
        "label": payload.label,
        "model": "SAM2.1 Hiera Small",
        "checkpoint": CHECKPOINT.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "points": [p.model_dump() for p in payload.points],
        "box": payload.box.model_dump() if payload.box else None,
        "source": payload.source.model_dump() if payload.source else None,
        **result,
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {**metadata, "mask_url": f"/workbench/api/vision/masks/{run_id}"}


@router.post("/ground")
def ground(payload: GroundIn):
    required = ("config.json", "preprocessor_config.json", "processing_florence2.py",
                "configuration_florence2.py", "modeling_florence2.py", "model.safetensors")
    dino = payload.detector == "grounding-dino-tiny"
    if dino and not _dino_ready():
        raise HTTPException(503, "Grounding DINO assets are not ready")
    if not TORCH_PY.is_file() or not GROUND_SCRIPT.is_file() or (not dino and any(not (FLORENCE_DIR / name).is_file() for name in required)):
        raise HTTPException(503, "Florence-2 model is not ready")
    raw = payload.image_base64.split(",", 1)[-1]
    try:
        image_bytes = base64.b64decode(raw, validate=True)
    except Exception as exc:
        raise HTTPException(400, "invalid image data") from exc

    run_id = str(uuid.uuid4())
    out_dir = VISION_DIR / "grounding" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / "image.png"
    result_path = out_dir / "result.json"
    request_path = out_dir / "request.json"
    image_path.write_bytes(image_bytes)
    request_path.write_text(json.dumps({
        "image": str(image_path), "model_dir": str(DINO_DIR if dino else FLORENCE_DIR),
        "detector": payload.detector, "text_threshold": payload.text_threshold,
        "result": str(result_path), "prompt": payload.prompt, "threshold": payload.threshold,
    }), encoding="utf-8")

    env = os.environ.copy()
    env.update({"PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    try:
        from services.vision_runtime import runtime
        request = json.loads(request_path.read_text(encoding="utf-8"))
        result = runtime.execute({**request, "operation": "ground"}, TORCH_PY, SCRIPT,
                                 settings.root / "vendor" / "sam2", settings.data_root,
                                 owner=payload.session_id or "api")
        result_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        raise HTTPException(503, str(exc)) from exc
    result.update({
        "id": run_id, "filename": Path(payload.filename).name,
        "source": payload.source.model_dump() if payload.source else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    (out_dir / "metadata.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _track_dir(run_id: str) -> Path:
    try:
        safe_id = str(uuid.UUID(run_id))
    except ValueError as exc:
        raise HTTPException(404, "track not found") from exc
    return VISION_DIR / "tracks" / safe_id


def _track_batch_dir(batch_id: str) -> Path:
    try:
        safe_id = str(uuid.UUID(batch_id))
    except ValueError as exc:
        raise HTTPException(404, "track batch not found") from exc
    return VISION_DIR / "batches" / safe_id


def _ground_batch_dir(batch_id: str) -> Path:
    try:
        safe_id = str(uuid.UUID(batch_id))
    except ValueError as exc:
        raise HTTPException(404, "ground batch not found") from exc
    return VISION_DIR / "ground-batches" / safe_id


def _probe_video_duration(video: Path) -> float:
    if not shutil.which("ffprobe"):
        raise HTTPException(503, "ffprobe is required for Episode sampling")
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
            capture_output=True, text=True, timeout=15, check=True,
        )
        duration = float(result.stdout.strip())
    except (subprocess.SubprocessError, ValueError) as exc:
        raise HTTPException(400, f"cannot read video duration: {video.name}") from exc
    if duration <= 0:
        raise HTTPException(400, f"video is empty: {video.name}")
    return duration


@router.post("/ground-batches", status_code=201)
def create_ground_batch(payload: GroundBatchIn):
    if payload.detector == "grounding-dino-tiny" and not _dino_ready():
        raise HTTPException(503, "Grounding DINO assets are not ready")
    if not TORCH_PY.is_file() or not GROUND_BATCH_SCRIPT.is_file() or not CHECKPOINT.is_file():
        raise HTTPException(503, "Florence-2/SAM2 batch runtime is not ready")
    if shutil.which("ffmpeg") is None:
        raise HTTPException(503, "ffmpeg is required for video frame sampling")

    from apps.api.routers.workbench import _dataset_path, _dataset_video_ref, _episode_indices

    dataset, version, _root = _dataset_path(payload.dataset_version_id)
    allowed = _episode_indices(version)
    if allowed is None:
        allowed = list(range(max(0, int(version.episode_count or 0)))) or None
    episodes = list(dict.fromkeys(int(ep) for ep in payload.episodes))
    if any(ep < 0 for ep in episodes):
        raise HTTPException(422, "episode must be non-negative")
    if allowed is not None:
        unknown = [ep for ep in episodes if ep not in set(allowed)]
        if unknown:
            raise HTTPException(404, f"Episode {unknown[0]} does not belong to this dataset version")

    plans: list[dict] = []
    skipped: list[dict] = []
    estimated_frames = 0
    for episode in episodes:
        try:
            video, segment_start, segment_end = _dataset_video_ref(
                payload.dataset_version_id, episode, payload.camera
            )
            origin = float(segment_start or 0.0)
            absolute_end = float(segment_end) if segment_end is not None else _probe_video_duration(video)
            duration = max(0.0, absolute_end - origin)
            start_rel = float(payload.start_s)
            end_rel = min(float(payload.end_s) if payload.end_s is not None else duration, duration)
            if start_rel >= duration or end_rel <= start_rel:
                skipped.append({"episode": episode, "status": "skipped", "reason": "采样时间超出视频范围"})
                continue
            estimate = max(1, math.ceil((end_rel - start_rel) / payload.interval_s))
            estimated_frames += estimate
            plans.append({
                "episode": episode,
                "video": str(video),
                "episode_start_s": origin,
                "sample_start_s": origin + start_rel,
                "sample_end_s": origin + end_rel,
                "estimated_frames": estimate,
                "duration_s": duration,
            })
        except HTTPException as exc:
            skipped.append({"episode": episode, "status": "skipped", "reason": str(exc.detail)})

    if not plans:
        raise HTTPException(400, "没有可处理的 Episode 视频")
    if estimated_frames > 120:
        raise HTTPException(422, f"本批预计 {estimated_frames} 帧；单批最多 120 帧，请缩短范围或增大抽帧间隔")

    batch_id = str(uuid.uuid4())
    batch_dir = VISION_DIR / "ground-batches" / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    request_path = batch_dir / "request.json"
    metadata = {
        "id": batch_id,
        "dataset_version_id": payload.dataset_version_id,
        "dataset_name": dataset.name if dataset else "",
        "episodes": episodes,
        "camera": payload.camera,
        "prompt": payload.prompt,
        "threshold": payload.threshold,
        "detector": payload.detector,
        "text_threshold": payload.text_threshold,
        "interval_s": payload.interval_s,
        "start_s": payload.start_s,
        "end_s": payload.end_s,
        "estimated_frames": estimated_frames,
        "skipped": skipped,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "queued",
    }
    request = {
        "batch_id": batch_id,
        "dataset_version_id": payload.dataset_version_id,
        "camera": payload.camera,
        "prompt": payload.prompt,
        "threshold": payload.threshold,
        "detector": payload.detector,
        "text_threshold": payload.text_threshold,
        "interval_s": payload.interval_s,
        "model_dir": str(DINO_DIR if payload.detector == "grounding-dino-tiny" else FLORENCE_DIR),
        "checkpoint": str(CHECKPOINT),
        "output_dir": str(batch_dir),
        "estimated_frames": estimated_frames,
        "episodes": plans,
        "created_at": metadata["created_at"],
    }
    vendor = str(settings.root / "vendor" / "sam2")
    with session_scope() as session:
        job = jobsvc.create_job(
            session,
            kind="vision_ground",
            name=f"{'dino' if payload.detector == 'grounding-dino-tiny' else 'florence'}-sam2-ds{payload.dataset_version_id}-{batch_id[:8]}",
            spec={
                "command": [str(TORCH_PY), str(GROUND_BATCH_SCRIPT), str(request_path)],
                "cwd": str(settings.data_root),
                "env": {
                    "PYTHONPATH": vendor,
                    "PYTHONUNBUFFERED": "1",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                },
                "gpu_memory_mb": 12000,
                "vision_ground_batch_id": batch_id,
                "vision_ground_output_dir": str(batch_dir),
            },
            requested_gpus=1,
            priority=55,
            max_attempts=1,
            created_by="vision_ground_batch",
            dataset_version_id=payload.dataset_version_id,
        )
        request["job_id"] = job.id
        metadata["job_id"] = job.id
        request_path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
        (batch_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (batch_dir / "progress.json").write_text(json.dumps({
            "status": job.status,
            "total_frames": estimated_frames,
            "completed_frames": 0,
        }), encoding="utf-8")
        return {
            "id": batch_id,
            "job_id": job.id,
            "status": job.status,
            "total_frames": estimated_frames,
            "episodes": episodes,
            "skipped": skipped,
        }


@router.get("/ground-batches/{batch_id}")
def get_ground_batch(batch_id: str):
    batch_dir = _ground_batch_dir(batch_id)
    metadata_path = batch_dir / "metadata.json"
    if not metadata_path.is_file():
        raise HTTPException(404, "ground batch not found")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    progress_path = batch_dir / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {}
    results_path = batch_dir / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.is_file() else {}
    with session_scope() as session:
        job = session.get(Job, int(metadata.get("job_id") or 0))
        status = str(job.status) if job is not None else "failed"
        job_error = job.error if job is not None else "Job not found"
    detections = results.get("detections") or []
    rendered = []
    for item in detections[-500:]:
        rendered.append({
            **item,
            "frame_url": f"/workbench/api/vision/ground-batches/{batch_id}/frames/{item['episode']}/{item['frame_index']}",
            "mask_url": f"/workbench/api/vision/ground-batches/{batch_id}/masks/{item['episode']}/{item['frame_index']}/{item['detection_index']}",
            "overlay_url": f"/workbench/api/vision/ground-batches/{batch_id}/overlays/{item['episode']}/{item['frame_index']}",
        })
    rendered_frames = []
    for item in (results.get("frames") or [])[-500:]:
        frame = {
            **{key: item.get(key) for key in ("episode", "camera", "frame_index", "timestamp_s", "filtered_low_confidence")},
            "detection_count": len(item.get("detections") or []),
            "frame_url": f"/workbench/api/vision/ground-batches/{batch_id}/frames/{item['episode']}/{item['frame_index']}",
        }
        if item.get("relative_overlay_path"):
            frame["overlay_url"] = f"/workbench/api/vision/ground-batches/{batch_id}/overlays/{item['episode']}/{item['frame_index']}"
        rendered_frames.append(frame)
    if progress.get("status") == "failed":
        status = "failed"
    response = {
        **metadata,
        "status": status,
        "progress": progress,
        "detections": rendered,
        "frames": rendered_frames,
        "result_count": len(detections),
        "completed_frames": int(results.get("completed_frames") or progress.get("completed_frames") or 0),
        "filtered_low_confidence": int(results.get("filtered_low_confidence") or 0),
        "elapsed_s": results.get("elapsed_s"),
        "device": results.get("device"),
        "error": progress.get("error") or (job_error if status == "failed" else None),
    }
    if status == "completed":
        response["download_url"] = f"/workbench/api/vision/ground-batches/{batch_id}/download"
    return response


@router.post("/ground-batches/{batch_id}/cancel")
def cancel_ground_batch(batch_id: str):
    batch_dir = _ground_batch_dir(batch_id)
    metadata_path = batch_dir / "metadata.json"
    if not metadata_path.is_file():
        raise HTTPException(404, "ground batch not found")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with session_scope() as session:
        if not jobsvc.cancel(session, int(metadata.get("job_id") or 0)):
            raise HTTPException(409, "ground batch is already finished")
    return {"id": batch_id, "job_id": metadata.get("job_id"), "status": "cancelled"}


@router.get("/ground-batches/{batch_id}/frames/{episode}/{frame_index}")
def get_ground_batch_frame(batch_id: str, episode: int, frame_index: int):
    path = _ground_batch_dir(batch_id) / f"episode_{episode:06d}" / "frames" / f"frame_{frame_index:05d}.jpg"
    if not path.is_file():
        raise HTTPException(404, "sample frame not found")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/ground-batches/{batch_id}/masks/{episode}/{frame_index}/{detection_index}")
def get_ground_batch_mask(batch_id: str, episode: int, frame_index: int, detection_index: int):
    path = _ground_batch_dir(batch_id) / f"episode_{episode:06d}" / "masks" / f"frame_{frame_index:05d}_det_{detection_index:02d}.png"
    if not path.is_file():
        raise HTTPException(404, "sample mask not found")
    return FileResponse(path, media_type="image/png")


@router.get("/ground-batches/{batch_id}/overlays/{episode}/{frame_index}")
def get_ground_batch_overlay(batch_id: str, episode: int, frame_index: int):
    path = _ground_batch_dir(batch_id) / f"episode_{episode:06d}" / "overlays" / f"frame_{frame_index:05d}.jpg"
    if not path.is_file():
        raise HTTPException(404, "sample overlay not found")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/ground-batches/{batch_id}/download")
def download_ground_batch(batch_id: str):
    batch_dir = _ground_batch_dir(batch_id)
    metadata_path = batch_dir / "metadata.json"
    if not metadata_path.is_file():
        raise HTTPException(404, "ground batch not found")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with session_scope() as session:
        job = session.get(Job, int(metadata.get("job_id") or 0))
        if job is None or job.status != "completed":
            raise HTTPException(409, "ground batch is not completed")
    archive = batch_dir / "annotation-export.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in batch_dir.rglob("*"):
            if not path.is_file() or path.name in {"request.json", "annotation-export.zip"}:
                continue
            if path.name in {"metadata.json", "progress.json", "results.json"} or path.suffix.lower() in {".jpg", ".png"}:
                bundle.write(path, path.relative_to(batch_dir).as_posix())
    return FileResponse(archive, media_type="application/zip", filename=f"florence_sam2_batch_{batch_id[:8]}.zip")


def _annotation_run_view(run: AnnotationRun) -> dict:
    return {
        "id": run.id,
        "dataset_version_id": run.dataset_version_id,
        "name": run.model_version,
        "kind": run.kind,
        "status": run.status,
        "episode_count": run.episode_count,
        "annotation_count": run.annotation_count,
        "created_at": run.created_at,
    }


@router.get("/annotation-runs")
def list_vision_annotation_runs(dataset_version_id: int):
    with session_scope() as session:
        rows = session.scalars(
            select(AnnotationRun)
            .where(AnnotationRun.dataset_version_id == dataset_version_id,
                   AnnotationRun.kind == "bbox_gt")
            .order_by(AnnotationRun.id.desc())
        ).all()
        return [_annotation_run_view(row) for row in rows]


@router.post("/annotation-runs", status_code=201)
def create_vision_annotation_run(payload: AnnotationRunIn):
    name = payload.name.strip()
    if not name:
        raise HTTPException(422, "annotation set name is required")
    with session_scope() as session:
        if session.get(DatasetVersion, payload.dataset_version_id) is None:
            raise HTTPException(404, "dataset version not found")
        run = AnnotationRun(
            dataset_version_id=payload.dataset_version_id,
            model_name="human",
            model_version=name,
            spec_version="bbox-v1",
            kind="bbox_gt",
            status="draft",
            episode_count=0,
            annotation_count=0,
        )
        session.add(run)
        session.flush()
        return _annotation_run_view(run)


@router.get("/annotation-runs/{run_id}/annotations")
def list_vision_annotations(run_id: int):
    with session_scope() as session:
        run = session.get(AnnotationRun, run_id)
        if run is None or run.kind != "bbox_gt":
            raise HTTPException(404, "vision annotation set not found")
        rows = session.scalars(
            select(Annotation).where(Annotation.run_id == run_id).order_by(Annotation.id)
        ).all()
        return {"run": _annotation_run_view(run), "annotations": [
            {"id": row.id, "kind": row.kind, "payload": row.payload or {},
             "reviewer": row.reviewer, "status": row.status}
            for row in rows
        ]}


@router.put("/annotation-runs/{run_id}/frames")
def save_vision_ground_truth_frame(run_id: int, payload: GroundTruthFrameIn):
    batch_dir = _ground_batch_dir(payload.batch_id)
    metadata_path = batch_dir / "metadata.json"
    results_path = batch_dir / "results.json"
    if not metadata_path.is_file() or not results_path.is_file():
        raise HTTPException(404, "grounding batch not found")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if results.get("status") != "completed":
        raise HTTPException(409, "grounding batch is not completed")
    if int(metadata.get("dataset_version_id") or 0) <= 0:
        raise HTTPException(400, "grounding batch has no dataset source")
    frame = next((row for row in results.get("frames", [])
                  if int(row.get("episode", -1)) == payload.episode
                  and row.get("camera") == payload.camera
                  and int(row.get("frame_index", -1)) == payload.frame_index), None)
    if frame is None or abs(float(frame.get("timestamp_s") or 0.0) - payload.timestamp_s) > 0.02:
        raise HTTPException(404, "source frame does not match this batch")
    for item in payload.boxes:
        box = item.box
        if not item.label.strip():
            raise HTTPException(422, "box label is required")
        if (box.x1 < 0 or box.y1 < 0 or box.x2 <= box.x1 or box.y2 <= box.y1
                or box.x2 > payload.width or box.y2 > payload.height):
            raise HTTPException(422, "box coordinates are outside the source image")

    with session_scope() as session:
        run = session.get(AnnotationRun, run_id)
        if run is None or run.kind != "bbox_gt":
            raise HTTPException(404, "vision annotation set not found")
        if run.status != "draft":
            raise HTTPException(409, "published annotation sets are read-only")
        version_id = int(metadata["dataset_version_id"])
        if int(run.dataset_version_id) != version_id:
            raise HTTPException(409, "annotation set and batch use different dataset versions")

        rows = session.scalars(select(Annotation).where(Annotation.run_id == run_id)).all()
        for old in rows:
            old_payload = old.payload or {}
            if (old_payload.get("batch_id") == payload.batch_id
                    and int(old_payload.get("episode", -1)) == payload.episode
                    and old_payload.get("camera") == payload.camera
                    and int(old_payload.get("frame_index", -1)) == payload.frame_index):
                session.delete(old)
        frame_ref = {
            "batch_id": payload.batch_id,
            "dataset_version_id": version_id,
            "episode": payload.episode,
            "camera": payload.camera,
            "frame_index": payload.frame_index,
            "timestamp_s": payload.timestamp_s,
            "width": payload.width,
            "height": payload.height,
            "complete": True,
        }
        session.add(Annotation(
            run_id=run.id, dataset_version_id=version_id,
            episode_index=payload.episode, kind="frame_gt", payload=frame_ref,
            confidence=None, model_version=run.model_version, spec_version=run.spec_version,
            status="accepted", reviewer=payload.reviewer.strip() or None,
        ))
        for item in payload.boxes:
            session.add(Annotation(
                run_id=run.id, dataset_version_id=version_id,
                episode_index=payload.episode, kind="bbox_gt",
                payload={**frame_ref, "label": item.label.strip(), "box": item.box.model_dump()},
                confidence=None, model_version=run.model_version, spec_version=run.spec_version,
                status="accepted", reviewer=payload.reviewer.strip() or None,
            ))
        session.flush()
        updated_rows = session.scalars(select(Annotation).where(Annotation.run_id == run_id)).all()
        run.annotation_count = sum(1 for row in updated_rows if row.kind == "bbox_gt")
        run.episode_count = len({int(row.episode_index) for row in updated_rows if row.kind == "frame_gt"})
        return {
            "run_id": run.id,
            "frame": frame_ref,
            "boxes_saved": len(payload.boxes),
            "frame_marked_complete": True,
            "annotation_count": run.annotation_count,
        }


@router.post("/annotation-runs/{run_id}/publish")
def publish_vision_annotation_run(run_id: int):
    with session_scope() as session:
        run = session.get(AnnotationRun, run_id)
        if run is None or run.kind != "bbox_gt":
            raise HTTPException(404, "vision annotation set not found")
        frames = session.scalars(
            select(Annotation).where(Annotation.run_id == run_id, Annotation.kind == "frame_gt")
        ).all()
        if not frames:
            raise HTTPException(409, "mark at least one frame complete before publishing")
        run.status = "ready"
        return _annotation_run_view(run)


def _box_iou(a: dict, b: dict) -> float:
    x1, y1 = max(float(a["x1"]), float(b["x1"])), max(float(a["y1"]), float(b["y1"]))
    x2, y2 = min(float(a["x2"]), float(b["x2"])), min(float(a["y2"]), float(b["y2"]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a["x2"]) - float(a["x1"])) * max(0.0, float(a["y2"]) - float(a["y1"]))
    area_b = max(0.0, float(b["x2"]) - float(b["x1"])) * max(0.0, float(b["y2"]) - float(b["y1"]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _ap101(predictions: list[dict], ground_truth: dict[tuple, list[dict]], label: str, threshold: float) -> float | None:
    gt_count = sum(1 for rows in ground_truth.values() for row in rows if row["label"] == label)
    if not gt_count:
        return None
    selected = sorted((row for row in predictions if row["label"] == label),
                      key=lambda row: float(row.get("score") or 0.0), reverse=True)
    matched: set[tuple[int, str, int, int]] = set()
    tp_flags, fp_flags = [], []
    for pred in selected:
        key = pred["frame_key"]
        choices = [(index, _box_iou(pred["box"], gt["box"]))
                   for index, gt in enumerate(ground_truth.get(key, []))
                   if gt["label"] == label and (key[0], key[1], key[2], index) not in matched]
        best = max(choices, key=lambda pair: pair[1], default=None)
        if best and best[1] >= threshold:
            matched.add((key[0], key[1], key[2], best[0]))
            tp_flags.append(1); fp_flags.append(0)
        else:
            tp_flags.append(0); fp_flags.append(1)
    cum_tp, cum_fp = 0, 0
    points = []
    for tp, fp in zip(tp_flags, fp_flags):
        cum_tp += tp; cum_fp += fp
        points.append((cum_tp / gt_count, cum_tp / max(1, cum_tp + cum_fp)))
    return sum(max((precision for recall, precision in points if recall >= target), default=0.0)
               for target in (index / 100 for index in range(101))) / 101


@router.post("/ground-batches/{batch_id}/evaluate")
def evaluate_ground_batch(batch_id: str, payload: GroundBenchmarkIn):
    batch_dir = _ground_batch_dir(batch_id)
    metadata_path, results_path = batch_dir / "metadata.json", batch_dir / "results.json"
    if not metadata_path.is_file() or not results_path.is_file():
        raise HTTPException(404, "grounding batch not found")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if results.get("status") != "completed":
        raise HTTPException(409, "grounding batch is not completed")
    with session_scope() as session:
        run = session.get(AnnotationRun, payload.annotation_run_id)
        if run is None or run.kind != "bbox_gt":
            raise HTTPException(404, "ground truth set not found")
        if run.status != "ready":
            raise HTTPException(409, "publish the ground truth set before evaluation")
        if int(run.dataset_version_id) != int(metadata.get("dataset_version_id") or 0):
            raise HTTPException(409, "ground truth set and batch use different dataset versions")
        rows = session.scalars(select(Annotation).where(Annotation.run_id == run.id)).all()
        gt_frames: set[tuple[int, str, int]] = set()
        ground_truth: dict[tuple[int, str, int], list[dict]] = {}
        for row in rows:
            item = row.payload or {}
            if item.get("batch_id") != batch_id:
                continue
            key = (int(item.get("episode", -1)), str(item.get("camera") or ""), int(item.get("frame_index", -1)))
            if row.kind == "frame_gt" and item.get("complete"):
                gt_frames.add(key)
                ground_truth.setdefault(key, [])
        for row in rows:
            item = row.payload or {}
            if item.get("batch_id") != batch_id or row.kind != "bbox_gt":
                continue
            key = (int(item.get("episode", -1)), str(item.get("camera") or ""), int(item.get("frame_index", -1)))
            if key in gt_frames:
                ground_truth.setdefault(key, []).append({"label": str(item.get("label") or "").strip().casefold(),
                                                         "box": item.get("box") or {}})
        if not gt_frames:
            raise HTTPException(422, "no completed source frames in this ground truth set")
        predictions = []
        for row in results.get("detections", []):
            key = (int(row.get("episode", -1)), str(row.get("camera") or ""), int(row.get("frame_index", -1)))
            if key not in gt_frames:
                continue
            predictions.append({"frame_key": key, "label": str(row.get("label") or "").strip().casefold(),
                                "box": row.get("box") or {}, "score": row.get("score")})

        classes = sorted({item["label"] for items in ground_truth.values() for item in items})
        matched: set[tuple[int, str, int, int]] = set()
        tp = fp = 0
        matched_ious = []
        for pred in sorted(predictions, key=lambda row: float(row.get("score") or 0.0), reverse=True):
            key = pred["frame_key"]
            candidates = [(index, _box_iou(pred["box"], gt["box"]))
                          for index, gt in enumerate(ground_truth.get(key, []))
                          if gt["label"] == pred["label"] and (*key, index) not in matched]
            best = max(candidates, key=lambda pair: pair[1], default=None)
            if best and best[1] >= payload.iou_threshold:
                matched.add((*key, best[0])); tp += 1; matched_ious.append(best[1])
            else:
                fp += 1
        gt_count = sum(len(items) for items in ground_truth.values())
        fn = max(0, gt_count - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / gt_count if gt_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_ap = {label: _ap101(predictions, ground_truth, label, payload.iou_threshold) for label in classes}
        map50 = sum(value for value in class_ap.values() if value is not None) / len(classes) if classes else None
        frame_total = len(results.get("frames") or [])
        metrics = {
            "protocol": "bbox IoU one-to-one matching; AP uses 101-point interpolation; only completed GT frames are scored",
            "iou_threshold": payload.iou_threshold,
            "dataset_version_id": int(run.dataset_version_id),
            "annotation_run_id": run.id,
            "annotation_set": run.model_version,
            "batch_id": batch_id,
            "model": "Florence-2-large",
            "mask_model": "SAM2.1 Hiera Small",
            "prompt": metadata.get("prompt"),
            "camera": metadata.get("camera"),
            "frames_total": frame_total,
            "frames_annotated": len(gt_frames),
            "frame_coverage": len(gt_frames) / frame_total if frame_total else 0.0,
            "ground_truth_boxes": gt_count,
            "predictions_scored": len(predictions),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "mean_matched_iou": sum(matched_ious) / len(matched_ious) if matched_ious else None,
            "ap50_per_class": class_ap,
            "map50": map50,
        }
        report_path = VISION_DIR / "benchmarks" / f"grounding_{batch_id}_{run.id}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        evaluation = Evaluation(source="vision", status="completed", metrics=metrics, report_uri=str(report_path))
        session.add(evaluation)
        session.flush()
        report = {"id": evaluation.id, "created_at": evaluation.created_at.isoformat(), "metrics": metrics}
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"id": evaluation.id, "metrics": metrics, "report_url": f"/workbench/api/vision/benchmarks/{evaluation.id}/report"}


@router.get("/benchmarks")
def list_vision_benchmarks(dataset_version_id: int | None = None):
    with session_scope() as session:
        rows = session.scalars(
            select(Evaluation).where(Evaluation.source == "vision", Evaluation.status == "completed")
            .order_by(Evaluation.id.desc()).limit(100)
        ).all()
        return [{"id": row.id, "created_at": row.created_at, "metrics": row.metrics or {},
                 "report_url": f"/workbench/api/vision/benchmarks/{row.id}/report"}
                for row in rows if dataset_version_id is None
                or int((row.metrics or {}).get("dataset_version_id", -1)) == dataset_version_id]


@router.get("/benchmarks/{evaluation_id}/report")
def get_vision_benchmark_report(evaluation_id: int):
    with session_scope() as session:
        evaluation = session.get(Evaluation, evaluation_id)
        if evaluation is None or evaluation.source != "vision" or not evaluation.report_uri:
            raise HTTPException(404, "vision benchmark not found")
        report_path = Path(evaluation.report_uri).resolve()
    root = (VISION_DIR / "benchmarks").resolve()
    if not report_path.is_relative_to(root) or not report_path.is_file():
        raise HTTPException(404, "vision benchmark report not found")
    return FileResponse(report_path, media_type="application/json", filename=report_path.name)


def _resolve_vision_mask_ref(mask_ref: str) -> dict:
    """Resolve an image Mask UUID or track-frame reference to its source files."""
    reference = str(mask_ref or "").strip()
    if reference.startswith("track:"):
        parts = reference.split(":")
        if len(parts) != 3:
            raise HTTPException(404, "video Mask reference is invalid")
        run_id = parts[1]
        try:
            frame_index = int(parts[2])
        except ValueError as exc:
            raise HTTPException(404, "video Mask frame is invalid") from exc
        run_dir = _track_dir(run_id)
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.is_file():
            raise HTTPException(404, "video track not found")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "completed":
            raise HTTPException(409, "video track is not completed")
        frame = next((row for row in metadata.get("frames", [])
                      if int(row.get("index", -1)) == frame_index), None)
        if frame is None:
            raise HTTPException(404, "video Mask frame not found")
        return {
            "id": f"track:{uuid.UUID(run_id)}:{frame_index}",
            "kind": "video",
            "run_id": str(uuid.UUID(run_id)),
            "frame_index": frame_index,
            "metadata_path": metadata_path,
            "metadata": metadata,
            "frame": frame,
            "source": metadata.get("source") or {},
            "mask_path": run_dir / "masks" / f"{frame_index:05d}.png",
            "frame_path": run_dir / "frames" / f"{frame_index:05d}.jpg",
            "overlay_path": run_dir / "overlays" / f"{frame_index:05d}.png",
            "filename": f"frame_{frame_index:05d}.png",
            "timestamp_s": float(frame.get("timestamp_s") or 0.0),
        }

    run_dir = _run_dir(reference)
    metadata_path = run_dir / "metadata.json"
    mask_path = run_dir / "mask.png"
    if not metadata_path.is_file():
        raise HTTPException(404, "SAM2 Mask not found")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not mask_path.is_file():
        raise HTTPException(404, "SAM2 Mask file is missing")
    source = metadata.get("source") or {}
    return {
        "id": str(metadata.get("id") or run_dir.name),
        "kind": "image",
        "run_id": str(metadata.get("id") or run_dir.name),
        "frame_index": None,
        "metadata_path": metadata_path,
        "metadata": metadata,
        "frame": None,
        "source": source,
        "mask_path": mask_path,
        "frame_path": run_dir / "image.png",
        "overlay_path": None,
        "filename": str(metadata.get("filename") or "mask.png"),
        "timestamp_s": float(source.get("timestamp_s") or 0.0),
    }


@router.post("/track", status_code=201)
def track_video(payload: TrackIn):
    if not payload.points or not any(p.label == 1 for p in payload.points):
        raise HTTPException(400, "add at least one positive point")
    if (not TORCH_PY.is_file() or not TRACK_SCRIPT.is_file() or not CHECKPOINT.is_file()
            or shutil.which("ffmpeg") is None):
        raise HTTPException(503, "SAM2 video runtime or checkpoint is not ready")

    from apps.api.routers.workbench import _dataset_video_ref

    source = payload.source
    video, segment_start, segment_end = _dataset_video_ref(
        source.dataset_version_id, source.episode, source.camera
    )
    seed_time = float(source.timestamp_s)
    segment_start = float(segment_start or 0.0)
    if seed_time < segment_start - 0.05:
        raise HTTPException(400, "selected frame is outside the Episode video")
    if segment_end is not None and seed_time > float(segment_end) + 0.05:
        raise HTTPException(400, "selected frame is outside the Episode video")

    run_id = str(uuid.uuid4())
    out_dir = VISION_DIR / "tracks" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    request_path = out_dir / "request.json"
    request = {
        "video": str(video),
        "output_dir": str(out_dir),
        "checkpoint": str(CHECKPOINT),
        "seed_timestamp_s": seed_time,
        "segment_end_s": float(segment_end) if segment_end is not None else None,
        "points": [point.model_dump() for point in payload.points],
        "source": source.model_dump(),
        "fps": 4,
        "window_seconds": 30,
    }
    vendor = str(settings.root / "vendor" / "sam2")
    with session_scope() as session:
        job = jobsvc.create_job(
            session,
            kind="vision_track",
            name=f"sam2-track-ds{source.dataset_version_id}-ep{source.episode}",
            spec={
                "command": [str(TORCH_PY), str(TRACK_SCRIPT), str(request_path)],
                "cwd": str(settings.data_root),
                "env": {"PYTHONPATH": vendor, "PYTHONUNBUFFERED": "1"},
                "gpu_memory_mb": 8192,
                "vision_track_id": run_id,
                "vision_track_output_dir": str(out_dir),
            },
            requested_gpus=1,
            priority=50,
            max_attempts=1,
            created_by="vision_sam2",
            dataset_version_id=source.dataset_version_id,
        )
        request["job_id"] = job.id
        request_path.write_text(json.dumps(request), encoding="utf-8")
        metadata = {
            "id": run_id,
            "job_id": job.id,
            "status": job.status,
            "model": "SAM2.1 Hiera Small Video Predictor",
            "checkpoint": CHECKPOINT.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": source.model_dump(),
            "points": [point.model_dump() for point in payload.points],
            "fps": 4,
            "window_seconds": 30,
            "frames": [],
        }
        (out_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return {key: metadata[key] for key in (
            "id", "job_id", "status", "model", "source", "fps", "window_seconds"
        )}


@router.post("/track-batches", status_code=201)
def track_video_batch(payload: TrackBatchIn):
    if not payload.points or not any(point.label == 1 for point in payload.points):
        raise HTTPException(400, "add at least one positive point")
    if (not TORCH_PY.is_file() or not TRACK_SCRIPT.is_file() or not CHECKPOINT.is_file()
            or shutil.which("ffmpeg") is None):
        raise HTTPException(503, "SAM2 video runtime or checkpoint is not ready")

    from apps.api.routers.workbench import (
        _dataset_path, _dataset_video_ref, _episode_indices,
    )

    source = payload.source
    _dataset, version, _root = _dataset_path(source.dataset_version_id)
    allowed = _episode_indices(version)
    if allowed is None:
        allowed = list(range(max(0, int(version.episode_count or 0))))
    allowed_set = set(allowed)
    if source.episode not in allowed_set:
        raise HTTPException(404, f"Episode {source.episode} does not belong to this dataset version")

    requested = list(dict.fromkeys(int(episode) for episode in payload.episodes))
    if any(episode < 0 for episode in requested):
        raise HTTPException(422, "episode must be non-negative")
    unknown = [episode for episode in requested if episode not in allowed_set]
    if unknown:
        raise HTTPException(404, f"Episode {unknown[0]} does not belong to this dataset version")

    _anchor_video, anchor_start, anchor_end = _dataset_video_ref(
        source.dataset_version_id, source.episode, source.camera
    )
    anchor_start = float(anchor_start or 0.0)
    anchor_time = float(source.timestamp_s)
    if anchor_time < anchor_start - 0.05 or (anchor_end is not None and anchor_time > float(anchor_end) + 0.05):
        raise HTTPException(400, "selected frame is outside the anchor Episode video")
    relative_time = max(0.0, anchor_time - anchor_start)

    batch_id = str(uuid.uuid4())
    batch_dir = VISION_DIR / "batches" / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []
    candidates: list[dict] = []
    for episode in requested:
        try:
            video, segment_start, segment_end = _dataset_video_ref(
                source.dataset_version_id, episode, source.camera
            )
        except HTTPException as exc:
            items.append({"episode": episode, "status": "skipped", "reason": str(exc.detail)})
            continue
        segment_start = float(segment_start or 0.0)
        seed_time = segment_start + relative_time
        if segment_end is not None and seed_time > float(segment_end) + 0.05:
            items.append({
                "episode": episode,
                "status": "skipped",
                "reason": "video is shorter than the selected prompt time",
            })
            continue

        run_id = str(uuid.uuid4())
        out_dir = VISION_DIR / "tracks" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        child_source = {
            "dataset_version_id": source.dataset_version_id,
            "episode": episode,
            "camera": source.camera,
            "timestamp_s": seed_time,
        }
        request = {
            "video": str(video),
            "output_dir": str(out_dir),
            "checkpoint": str(CHECKPOINT),
            "seed_timestamp_s": seed_time,
            "segment_end_s": float(segment_end) if segment_end is not None else None,
            "points": [point.model_dump() for point in payload.points],
            "source": child_source,
            "fps": 4,
            "window_seconds": 30,
        }
        candidates.append({
            "episode": episode,
            "run_id": run_id,
            "out_dir": out_dir,
            "request": request,
            "source": child_source,
        })

    if not candidates:
        raise HTTPException(400, "no selected Episode has video at the prompt time")

    vendor = str(settings.root / "vendor" / "sam2")
    with session_scope() as session:
        for candidate in candidates:
            request_path = candidate["out_dir"] / "request.json"
            source_row = candidate["source"]
            job = jobsvc.create_job(
                session,
                kind="vision_track",
                name=f"sam2-batch-ds{source.dataset_version_id}-ep{source_row['episode']}",
                spec={
                    "command": [str(TORCH_PY), str(TRACK_SCRIPT), str(request_path)],
                    "cwd": str(settings.data_root),
                    "env": {"PYTHONPATH": vendor, "PYTHONUNBUFFERED": "1"},
                    "gpu_memory_mb": 8192,
                    "vision_track_id": candidate["run_id"],
                    "vision_track_output_dir": str(candidate["out_dir"]),
                    "vision_track_batch_id": batch_id,
                },
                requested_gpus=1,
                priority=50,
                max_attempts=1,
                created_by="vision_sam2_batch",
                dataset_version_id=source.dataset_version_id,
            )
            candidate["request"]["job_id"] = job.id
            request_path.write_text(
                json.dumps(candidate["request"], ensure_ascii=False), encoding="utf-8"
            )
            (candidate["out_dir"] / "metadata.json").write_text(json.dumps({
                "id": candidate["run_id"],
                "job_id": job.id,
                "status": job.status,
                "model": "SAM2.1 Hiera Small Video Predictor",
                "checkpoint": CHECKPOINT.name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source": source_row,
                "points": [point.model_dump() for point in payload.points],
                "fps": 4,
                "window_seconds": 30,
                "frames": [],
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            items.append({
                "episode": source_row["episode"],
                "status": job.status,
                "job_id": job.id,
                "track_id": candidate["run_id"],
                "source": source_row,
            })

    metadata = {
        "id": batch_id,
        "dataset_version_id": source.dataset_version_id,
        "camera": source.camera,
        "anchor_episode": source.episode,
        "anchor_timestamp_s": anchor_time,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "points": [point.model_dump() for point in payload.points],
        "items": sorted(items, key=lambda item: int(item["episode"])),
    }
    (batch_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "id": batch_id,
        "dataset_version_id": source.dataset_version_id,
        "job_ids": [item["job_id"] for item in items if item.get("job_id")],
        "total": len(items),
        "queued": sum(item["status"] == "queued" for item in items),
        "skipped": sum(item["status"] == "skipped" for item in items),
        "status": "queued",
    }


@router.get("/track-batches/{batch_id}")
def get_track_video_batch(batch_id: str):
    path = _track_batch_dir(batch_id) / "metadata.json"
    if not path.is_file():
        raise HTTPException(404, "track batch not found")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    items = metadata.get("items") or []
    counts = {status: 0 for status in ("queued", "running", "completed", "failed", "cancelled", "skipped")}
    with session_scope() as session:
        for item in items:
            if item.get("status") == "skipped":
                counts["skipped"] += 1
                continue
            job = session.get(Job, int(item.get("job_id") or 0))
            status = str(job.status) if job is not None else "failed"
            if status not in counts:
                status = "failed"
            item["status"] = status
            counts[status] += 1
            if status == "completed" and item.get("track_id"):
                track_path = _track_dir(item["track_id"]) / "metadata.json"
                if track_path.is_file():
                    track = json.loads(track_path.read_text(encoding="utf-8"))
                    frames = track.get("frames") or []
                    item["frame_count"] = len(frames)
                    item["review_count"] = sum(bool(frame.get("review_suggested")) for frame in frames)
                    item["lost_frames"] = int(track.get("lost_frames") or 0)
                    item["track_url"] = f"/workbench/api/vision/tracks/{item['track_id']}"
                    item["download_url"] = f"/workbench/api/vision/tracks/{item['track_id']}/download"
                else:
                    item["status"] = "failed"
                    counts["completed"] -= 1
                    counts["failed"] += 1
    terminal = counts["queued"] == 0 and counts["running"] == 0
    metadata["progress"] = counts
    if terminal:
        metadata["status"] = "completed" if counts["completed"] or counts["skipped"] else "failed"
    else:
        metadata["status"] = "running" if counts["running"] or counts["completed"] or counts["failed"] else "queued"
    return metadata


@router.post("/tracks/{run_id}/correct", status_code=201)
def correct_track_keyframe(run_id: str, payload: TrackCorrectionIn):
    if not payload.points or not any(point.label == 1 for point in payload.points):
        raise HTTPException(400, "add at least one positive point")
    parent_dir = _track_dir(run_id)
    parent_file = parent_dir / "metadata.json"
    if not parent_file.is_file():
        raise HTTPException(404, "track not found")
    parent = json.loads(parent_file.read_text(encoding="utf-8"))
    if parent.get("status") != "completed":
        raise HTTPException(409, "only a completed track can be corrected")
    parent_source = parent.get("source") or {}
    source = payload.source
    if any([
        int(parent_source.get("dataset_version_id", -1)) != source.dataset_version_id,
        int(parent_source.get("episode", -1)) != source.episode,
        str(parent_source.get("camera", "")) != source.camera,
    ]):
        raise HTTPException(400, "correction source must match the selected track")
    frames = parent.get("frames") or []
    if not frames:
        raise HTTPException(409, "track has no frames to correct")
    fps = max(1, int(parent.get("fps") or 4))
    selected = min(frames, key=lambda frame: abs(float(frame["timestamp_s"]) - source.timestamp_s))
    if abs(float(selected["timestamp_s"]) - source.timestamp_s) > max(0.12, 0.6 / fps):
        raise HTTPException(400, "select a frame from the current track")
    correction_timestamp = float(selected["timestamp_s"])
    parent_last_timestamp = float(parent.get("end_timestamp_s") or correction_timestamp)
    from apps.api.routers.workbench import _dataset_video_ref
    video, _segment_start, video_end = _dataset_video_ref(
        source.dataset_version_id, source.episode, source.camera
    )
    exclusive_end = parent_last_timestamp + 1.0 / fps
    if video_end is not None:
        exclusive_end = min(exclusive_end, float(video_end))
    if correction_timestamp >= parent_last_timestamp - 0.001:
        raise HTTPException(400, "select a keyframe before the end of the tracked video")

    prefix_frames = [
        dict(frame) for frame in frames
        if int(frame.get("index", -1)) < int(selected.get("index", 0))
    ]
    correction = {
        "timestamp_s": correction_timestamp,
        "frame_index": int(selected["index"]),
        "points": [point.model_dump() for point in payload.points],
    }
    corrections = [*(parent.get("corrections") or []), correction]
    run_id_new = str(uuid.uuid4())
    out_dir = VISION_DIR / "tracks" / run_id_new
    out_dir.mkdir(parents=True, exist_ok=True)
    request_path = out_dir / "request.json"
    original_start = float(parent.get("start_timestamp_s") or parent_source.get("timestamp_s") or 0.0)
    requested_window = max(1, min(30, math.ceil(exclusive_end - correction_timestamp)))
    request = {
        "video": str(video),
        "output_dir": str(out_dir),
        "checkpoint": str(CHECKPOINT),
        "seed_timestamp_s": correction_timestamp,
        "segment_end_s": exclusive_end,
        "points": [point.model_dump() for point in payload.points],
        "parent_points": parent.get("points") or [],
        "source": parent_source,
        "fps": fps,
        "window_seconds": requested_window,
        "parent_run_id": run_id,
        "parent_run_dir": str(parent_dir),
        "prefix_frames": prefix_frames,
        "original_start_timestamp_s": original_start,
        "requested_window_s": parent.get("requested_window_s", parent.get("window_seconds", 30)),
        "corrections": corrections,
        "correction": correction,
    }
    vendor = str(settings.root / "vendor" / "sam2")
    with session_scope() as session:
        job = jobsvc.create_job(
            session,
            kind="vision_track",
            name=f"sam2-correct-ds{source.dataset_version_id}-ep{source.episode}",
            spec={
                "command": [str(TORCH_PY), str(TRACK_SCRIPT), str(request_path)],
                "cwd": str(settings.data_root),
                "env": {"PYTHONPATH": vendor, "PYTHONUNBUFFERED": "1"},
                "gpu_memory_mb": 8192,
                "vision_track_id": run_id_new,
                "vision_track_output_dir": str(out_dir),
                "vision_parent_track_id": run_id,
                "vision_correction_timestamp_s": correction_timestamp,
            },
            requested_gpus=1,
            priority=50,
            max_attempts=1,
            created_by="vision_sam2",
            dataset_version_id=source.dataset_version_id,
        )
        request["job_id"] = job.id
        request_path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
        metadata = {
            "id": run_id_new,
            "job_id": job.id,
            "status": job.status,
            "model": "SAM2.1 Hiera Small Video Predictor",
            "checkpoint": CHECKPOINT.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": parent_source,
            "points": parent.get("points") or [],
            "parent_run_id": run_id,
            "corrections": corrections,
            "correction": correction,
            "fps": fps,
            "window_seconds": requested_window,
            "frames": [],
        }
        (out_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return {
            "id": run_id_new,
            "job_id": job.id,
            "status": job.status,
            "parent_run_id": run_id,
            "correction": correction,
        }


def _track_response(run_id: str, metadata: dict) -> dict:
    frames = [{
        **frame,
        "frame_url": f"/workbench/api/vision/tracks/{run_id}/frames/{frame['index']}",
        "mask_url": f"/workbench/api/vision/tracks/{run_id}/masks/{frame['index']}",
        "overlay_url": f"/workbench/api/vision/tracks/{run_id}/overlays/{frame['index']}",
    } for frame in metadata.get("frames", [])]
    return {**metadata, "frames": frames,
            "download_url": f"/workbench/api/vision/tracks/{run_id}/download"}


@router.get("/tracks/{run_id}")
def get_track(run_id: str):
    path = _track_dir(run_id) / "metadata.json"
    if not path.is_file():
        raise HTTPException(404, "track not found")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("status") != "completed":
        return metadata
    return _track_response(run_id, metadata)


def _track_frame_path(run_id: str, frame_index: int, folder: str, suffix: str) -> Path:
    run_dir = _track_dir(run_id)
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.is_file():
        raise HTTPException(404, "track not found")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if frame_index < 0 or frame_index >= len(metadata.get("frames", [])):
        raise HTTPException(404, "track frame not found")
    path = run_dir / folder / f"{frame_index:05d}.{suffix}"
    if not path.is_file():
        raise HTTPException(404, "track frame not found")
    return path


@router.get("/tracks/{run_id}/frames/{frame_index}")
def get_track_frame(run_id: str, frame_index: int):
    return FileResponse(_track_frame_path(run_id, frame_index, "frames", "jpg"), media_type="image/jpeg")


@router.get("/tracks/{run_id}/masks/{frame_index}")
def get_track_mask(run_id: str, frame_index: int):
    return FileResponse(_track_frame_path(run_id, frame_index, "masks", "png"), media_type="image/png")


@router.get("/tracks/{run_id}/overlays/{frame_index}")
def get_track_overlay(run_id: str, frame_index: int):
    return FileResponse(_track_frame_path(run_id, frame_index, "overlays", "png"), media_type="image/png")


@router.get("/tracks/{run_id}/download")
def download_track(run_id: str):
    run_dir = _track_dir(run_id)
    metadata = run_dir / "metadata.json"
    if not metadata.is_file():
        raise HTTPException(404, "track not found")
    run_metadata = json.loads(metadata.read_text(encoding="utf-8"))
    if run_metadata.get("status") != "completed":
        raise HTTPException(409, f"track status is {run_metadata.get('status', 'unknown')}")
    archive = run_dir / "sam2_track.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(metadata, "metadata.json")
        for folder in ("frames", "masks", "overlays"):
            for file_path in sorted((run_dir / folder).glob("*")):
                zf.write(file_path, f"{folder}/{file_path.name}")
    return FileResponse(archive, media_type="application/zip", filename=f"sam2_track_{run_id[:8]}.zip")


@router.get("/masks/{run_id}")
def get_mask(run_id: str):
    path = _run_dir(run_id) / "mask.png"
    if not path.is_file():
        raise HTTPException(404, "mask not found")
    return FileResponse(path, media_type="image/png")


@router.get("/runs")
def list_image_masks(dataset_version_id: int | None = None, limit: int = 30):
    items = []
    for path in VISION_DIR.glob("*/metadata.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            if not (path.parent / "mask.png").is_file() or not row.get("id"):
                continue
            if dataset_version_id is not None and (row.get("source") or {}).get("dataset_version_id") != dataset_version_id:
                continue
            items.append({key: row.get(key) for key in ("id", "filename", "created_at", "source", "score")})
        except (OSError, json.JSONDecodeError):
            continue
    items.sort(key=lambda row: row.get("created_at") or "", reverse=True)
    return {"runs": items[:max(1, min(limit, 100))]}


@router.get("/runs/{run_id}/image")
def get_mask_source_image(run_id: str):
    path = _run_dir(run_id) / "image.png"
    if not path.is_file():
        raise HTTPException(404, "source image not found")
    return FileResponse(path, media_type="image/png")


@router.get("/runs/{run_id}")
def get_run(run_id: str):
    path = _run_dir(run_id) / "metadata.json"
    if not path.is_file():
        raise HTTPException(404, "run not found")
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/runs/{run_id}/coco")
def export_mask_coco(run_id: str):
    from PIL import Image
    run_dir = _run_dir(run_id)
    metadata = get_run(run_id)
    mask_path = run_dir / "mask.png"
    if not mask_path.is_file():
        raise HTTPException(404, "mask not found")
    with Image.open(mask_path) as source:
        mask = source.convert("L").point(lambda value: 255 if value else 0)
        width, height = mask.size
        pixels = mask.transpose(Image.Transpose.TRANSPOSE).tobytes()
        counts, previous, length, area = [], 0, 0, 0
        for pixel in pixels:
            value = int(pixel != 0)
            area += value
            if value == previous:
                length += 1
            else:
                counts.append(length)
                previous, length = value, 1
        counts.append(length)
        box = mask.getbbox()
    bbox = [box[0], box[1], box[2]-box[0], box[3]-box[1]] if box else [0, 0, 0, 0]
    document = {
        "info": {"description": "EmbodiedFlow SAM2 mask", "rle_encoding": "uncompressed column-major COCO RLE"},
        "images": [{"id": 1, "file_name": metadata.get("filename", "image.png"), "width": width, "height": height,
                    "source": metadata.get("source")}],
        "categories": [{"id": 1, "name": metadata.get("label", "object")}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "iscrowd": 0,
                         "segmentation": {"size": [height, width], "counts": counts},
                         "area": area, "bbox": bbox, "mask_run_id": run_id,
                         "mask_predicted_iou": metadata.get("score")}],
    }
    output = run_dir / "coco_rle.json"
    output.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return FileResponse(output, media_type="application/json", filename=f"sam2_{run_id[:8]}_coco.json")


def _flora_setup_state() -> str:
    try:
        return str(json.loads(FLORA_SETUP.read_text(encoding="utf-8")).get("state", "unknown"))
    except (OSError, json.JSONDecodeError):
        return "unavailable"


@router.get("/flora/status")
def flora_status():
    state = _flora_setup_state()
    return {
        "ready": state == "complete" and FLORA_ANNOTATE.is_file(),
        "state": state,
        "model": "NVIDIA FLORA",
        "profiles": ["yam", "dex", "tianji", "human"],
    }


@router.get("/flora/jobs")
def list_flora_jobs(dataset_version_id: int, limit: int = 30):
    with session_scope() as session:
        rows = session.scalars(select(Job).where(
            Job.kind == "vision_flora", Job.dataset_version_id == dataset_version_id
        ).order_by(Job.id.desc()).limit(max(1, min(limit, 100)))).all()
        return {"jobs": [{
            "job_id": row.id, "name": row.name, "status": row.status,
            "created_at": row.created_at.isoformat(),
            "profile": (row.spec or {}).get("flora_profile"),
            "episodes": (row.spec or {}).get("flora_episode_ids", []),
            "review_revision": (row.spec or {}).get("flora_review_revision", 0),
        } for row in rows]}


@router.post("/flora/jobs", status_code=201)
def create_flora_job(payload: FloraJobIn):
    if payload.profile not in {"yam", "dex", "tianji", "human"}:
        raise HTTPException(400, "FLORA profile must be yam, dex, tianji, or human")
    if len(payload.instruction) > 2000:
        raise HTTPException(400, "instruction is too long")
    state = _flora_setup_state()
    if state != "complete" or not FLORA_ANNOTATE.is_file():
        raise HTTPException(503, f"FLORA is not ready ({state})")

    from apps.api.routers.workbench import (
        _dataset_path, _dataset_video_ref, dataset_episodes,
    )

    _dataset, version, dataset_path = _dataset_path(payload.dataset_version_id)
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.is_file():
        raise HTTPException(400, "dataset has no LeRobot video manifest")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    camera_keys = {
        key.removeprefix("observation.images.")
        for key in (info.get("features") or {})
        if key.startswith("observation.images.")
    }
    if payload.camera not in camera_keys:
        raise HTTPException(400, f"dataset has no camera {payload.camera}")

    if payload.episodes is not None:
        episode_ids = list(dict.fromkeys(payload.episodes))
        if any(index < 0 for index in episode_ids):
            raise HTTPException(422, "episode indices must be non-negative")
    elif payload.all_episodes:
        rows = dataset_episodes(payload.dataset_version_id, limit=max(1, int(version.episode_count or 1)))
        episode_ids = [int(row["index"]) for row in rows]
        if not episode_ids and version.episode_count:
            episode_ids = list(range(int(version.episode_count)))
    else:
        episode_ids = [int(payload.episode)]
    if not episode_ids:
        raise HTTPException(400, "dataset has no episodes")

    left_key = next((k for k in ("left", "left_wrist", "wrist_left") if k in camera_keys), None)
    right_key = next((k for k in ("right", "right_wrist", "wrist_right") if k in camera_keys), None)
    add_wrist_views = (
        payload.profile != "human" and payload.camera in {"top", "front", "main"}
        and left_key and right_key
    )

    run_id = str(uuid.uuid4())
    out_dir = VISION_DIR / "flora" / run_id
    output_dir = out_dir / "annotations"
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs_file = out_dir / "jobs.jsonl"
    rows = []
    for episode_id in episode_ids:
        main_video, start, end = _dataset_video_ref(
            payload.dataset_version_id, episode_id, payload.camera
        )
        if end is not None:
            raise HTTPException(400, "FLORA requires one video file per Episode; select a LeRobot v2 dataset version")
        item = {
            "id": f"episode_{episode_id:06d}",
            "video": str(main_video),
            "profile": payload.profile,
        }
        if payload.instruction.strip():
            item["instruction"] = payload.instruction.strip()
        if add_wrist_views:
            for field, camera_key in (("left_wrist", left_key), ("right_wrist", right_key)):
                video, wrist_start, wrist_end = _dataset_video_ref(
                    payload.dataset_version_id, episode_id, camera_key
                )
                if wrist_end is not None or (start is not None and wrist_start != start):
                    raise HTTPException(400, "FLORA wrist videos must use one synchronized file per Episode")
                item[field] = str(video)
        rows.append(item)
    jobs_file.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    result_ids = [row["id"] for row in rows]
    command = [
        str(FLORA_ANNOTATE), "--input-jsonl", str(jobs_file),
        "--output-dir", str(output_dir),
    ]
    if add_wrist_views:
        profile_group = "三相机"
    else:
        profile_group = "单相机"
    spec = {
        "command": command,
        "cwd": str(FLORA_ROOT),
        "exp_dir": str(out_dir),
        "flora_output_dir": str(output_dir),
        "flora_run_id": run_id,
        "flora_dataset_version_id": payload.dataset_version_id,
        "flora_episode_ids": episode_ids,
        "flora_result_ids": result_ids,
        "flora_camera": payload.camera,
        "flora_profile": payload.profile,
        "flora_camera_group": profile_group,
        "gpu_memory_mb": 16000,
    }
    with session_scope() as session:
        job = jobsvc.create_job(
            session, kind="vision_flora",
            name=f"flora-{payload.profile}-ds{payload.dataset_version_id}-{len(episode_ids)}ep",
            spec=spec, requested_gpus=3, priority=50, max_attempts=1,
            created_by="vision_flora", dataset_version_id=payload.dataset_version_id,
        )
        return {
            "job_id": job.id, "status": job.status,
            "episodes": episode_ids, "camera": payload.camera,
            "profile": payload.profile, "camera_group": profile_group,
        }


@router.get("/flora/jobs/{job_id}/result")
def flora_job_result(job_id: int):
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None or job.kind != "vision_flora":
            raise HTTPException(404, "FLORA job not found")
        if job.status != "completed":
            raise HTTPException(409, f"FLORA job status is {job.status}")
        spec = dict(job.spec or {})
    output_dir = Path(spec.get("flora_output_dir", ""))
    annotations = []
    for episode_id, result_id in zip(
        spec.get("flora_episode_ids", []), spec.get("flora_result_ids", []), strict=True
    ):
        path = output_dir / f"{result_id}.json"
        if not path.is_file():
            raise HTTPException(404, f"FLORA result for Episode {episode_id} is missing")
        annotations.append({
            "dataset_version_id": spec.get("flora_dataset_version_id"),
            "episode": int(episode_id),
            "camera": spec.get("flora_camera"),
            "profile": spec.get("flora_profile"),
            "annotation": json.loads(path.read_text(encoding="utf-8")),
        })
    return {"job_id": job_id, "annotations": annotations}


@router.get("/flora/jobs/{job_id}/masks")
def flora_job_masks(job_id: int):
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None or job.kind != "vision_flora":
            raise HTTPException(404, "FLORA job not found")
        spec = dict(job.spec or {})
    version_id = int(spec.get("flora_dataset_version_id", 0))
    episodes = {int(value) for value in spec.get("flora_episode_ids", [])}
    runs = []
    for metadata_path in VISION_DIR.glob("*/metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            source = metadata.get("source") or {}
            if (int(source.get("dataset_version_id", -1)) != version_id
                    or int(source.get("episode", -1)) not in episodes
                    or not (metadata_path.parent / "mask.png").is_file()):
                continue
            mask_url = f"/workbench/api/vision/masks/{metadata.get('id') or metadata_path.parent.name}"
            runs.append({
                "id": str(metadata.get("id") or metadata_path.parent.name),
                "kind": "image",
                "episode": int(source["episode"]),
                "camera": source.get("camera", ""),
                "timestamp_s": float(source.get("timestamp_s", 0)),
                "filename": metadata.get("filename", "mask.png"),
                "score": metadata.get("score"),
                "mask_url": mask_url,
                "preview_url": mask_url,
            })
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue

    track_rows = []
    superseded_tracks = set()
    for metadata_path in (VISION_DIR / "tracks").glob("*/metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            source = metadata.get("source") or {}
            episode = int(source.get("episode", -1))
            if (metadata.get("status") != "completed"
                    or int(source.get("dataset_version_id", -1)) != version_id
                    or episode not in episodes):
                continue
            run_id = str(metadata.get("id") or metadata_path.parent.name)
            track_rows.append((run_id, metadata, source, metadata_path.parent))
            parent_id = metadata.get("parent_run_id")
            if parent_id:
                superseded_tracks.add(str(parent_id))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue

    for run_id, metadata, source, run_dir in track_rows:
        if run_id in superseded_tracks:
            continue
        frames = metadata.get("frames") or []
        sample_stride = max(1, int(metadata.get("fps") or 4))
        for position, frame in enumerate(frames):
            try:
                frame_index = int(frame.get("index", position))
                mask_path = run_dir / "masks" / f"{frame_index:05d}.png"
                if not mask_path.is_file():
                    continue
                review_suggested = bool(frame.get("review_suggested"))
                if frame_index % sample_stride and not review_suggested and position != len(frames) - 1:
                    continue
                runs.append({
                    "id": f"track:{run_id}:{frame_index}",
                    "kind": "video",
                    "track_id": run_id,
                    "frame_index": frame_index,
                    "episode": int(source["episode"]),
                    "camera": source.get("camera", ""),
                    "timestamp_s": float(frame.get("timestamp_s", 0)),
                    "filename": f"frame_{frame_index:05d}.png",
                    "review_suggested": review_suggested,
                    "mask_url": f"/workbench/api/vision/tracks/{run_id}/masks/{frame_index}",
                    "preview_url": f"/workbench/api/vision/tracks/{run_id}/overlays/{frame_index}",
                })
            except (OSError, ValueError, TypeError):
                continue
    runs.sort(key=lambda row: (row["episode"], row["timestamp_s"], row["camera"]))
    return {"job_id": job_id, "masks": runs}


@router.get("/flora/jobs/{job_id}/review")
def flora_job_review(job_id: int, revision: int | None = None):
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None or job.kind != "vision_flora":
            raise HTTPException(404, "FLORA job not found")
        spec = dict(job.spec or {})
    if revision is not None:
        if revision < 1:
            raise HTTPException(422, "revision must be positive")
        review_path = Path(spec.get("exp_dir", "")) / "reviewed" / "revisions" / f"r{revision:04d}" / "annotations.json"
    else:
        review_path = Path(spec.get("flora_review_path", ""))
    if not review_path.is_file():
        legacy_path = Path(spec.get("flora_review_path", ""))
        if revision == 1 and legacy_path.is_file():
            legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
            if int(legacy.get("revision") or 1) == 1:
                return legacy
        raise HTTPException(404, "FLORA review revision not found")
    return json.loads(review_path.read_text(encoding="utf-8"))


@router.get("/flora/jobs/{job_id}/review-versions")
def flora_job_review_versions(job_id: int):
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None or job.kind != "vision_flora":
            raise HTTPException(404, "FLORA job not found")
        spec = dict(job.spec or {})
    review_root = Path(spec.get("exp_dir", "")) / "reviewed"
    paths = sorted((review_root / "revisions").glob("r[0-9]*/annotations.json"))
    latest_path = Path(spec.get("flora_review_path", ""))
    if latest_path.is_file() and latest_path not in paths:
        paths.append(latest_path)
    versions = []
    for path in paths:
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        versions.append({
            "revision": int(item.get("revision") or 1),
            "reviewed_at": item.get("reviewed_at"),
            "segments": sum(len(row.get("segments", [])) for row in item.get("annotations", [])),
            "episodes": len(item.get("annotations", [])),
        })
    versions.sort(key=lambda row: row["revision"])
    return {"job_id": job_id, "current_revision": int(spec.get("flora_review_revision") or (versions[-1]["revision"] if versions else 0)), "versions": versions}


@router.get("/flora/jobs/{job_id}/review-diff")
def flora_review_diff(job_id: int, from_revision: int, to_revision: int):
    before = flora_job_review(job_id, from_revision)
    after = flora_job_review(job_id, to_revision)
    left = {int(item["episode"]): item for item in before.get("annotations", [])}
    right = {int(item["episode"]): item for item in after.get("annotations", [])}
    changes = []
    for episode in sorted(set(left) | set(right)):
        old, new = left.get(episode, {}), right.get(episode, {})
        old_segments, new_segments = old.get("segments", []), new.get("segments", [])
        segment_changes = []
        for index in range(max(len(old_segments), len(new_segments))):
            a = old_segments[index] if index < len(old_segments) else None
            b = new_segments[index] if index < len(new_segments) else None
            if a != b:
                segment_changes.append({"index": index, "before": a, "after": b,
                                        "kind": "added" if a is None else "removed" if b is None else "changed"})
        removed_masks = sorted(set(old.get("mask_run_ids", [])) - set(new.get("mask_run_ids", [])))
        added_masks = sorted(set(new.get("mask_run_ids", [])) - set(old.get("mask_run_ids", [])))
        if segment_changes or removed_masks or added_masks:
            changes.append({"episode": episode, "segments": segment_changes,
                            "masks_added": added_masks, "masks_removed": removed_masks})
    return {"job_id": job_id, "from_revision": from_revision, "to_revision": to_revision,
            "changed_episodes": len(changes), "changes": changes,
            "notes": {"before": before.get("notes", ""), "after": after.get("notes", "")},
            "comparison": "segment position; inserted/deleted rows can shift later indices"}


@router.post("/flora/jobs/{job_id}/review")
def save_flora_review(job_id: int, payload: FloraReviewIn):
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None or job.kind != "vision_flora":
            raise HTTPException(404, "FLORA job not found")
        if job.status != "completed":
            raise HTTPException(409, f"FLORA job status is {job.status}")
        spec = dict(job.spec or {})
        allowed_episodes = {int(value) for value in spec.get("flora_episode_ids", [])}
        if not payload.annotations:
            raise HTTPException(400, "至少保留一个 Episode")
        reviewed_episodes = set()
        annotations = []
        for item in payload.annotations:
            episode = int(item.episode)
            if episode not in allowed_episodes or episode in reviewed_episodes:
                raise HTTPException(400, f"Episode {episode} 不属于此任务或重复")
            reviewed_episodes.add(episode)
            segments = []
            for segment in item.segments:
                caption = segment.caption.strip()
                if segment.start_s < 0 or segment.end_s <= segment.start_s or not caption:
                    raise HTTPException(400, f"Episode {episode} 的片段时间或描述无效")
                segments.append({
                    "start_s": round(segment.start_s, 3),
                    "end_s": round(segment.end_s, 3),
                    "caption": caption,
                })
            mask_run_ids = []
            for mask_ref in dict.fromkeys(item.mask_run_ids):
                try:
                    mask = _resolve_vision_mask_ref(mask_ref)
                except HTTPException as exc:
                    raise HTTPException(400, f"SAM2 Mask {mask_ref} 无效：{exc.detail}") from exc
                source = mask.get("source") or {}
                if (int(source.get("dataset_version_id", -1)) != int(spec["flora_dataset_version_id"])
                        or int(source.get("episode", -1)) != episode):
                    raise HTTPException(400, f"SAM2 Mask {mask_ref} 与数据 Episode 不匹配")
                if not mask["mask_path"].is_file():
                    raise HTTPException(400, f"SAM2 Mask {mask_ref} 文件缺失")
                mask_run_ids.append(mask["id"])
            annotations.append({"episode": episode, "segments": segments, "mask_run_ids": mask_run_ids})

        reviewed_at = utcnow().isoformat()
        review_root = Path(spec.get("exp_dir", "")) / "reviewed"
        revisions_root = review_root / "revisions"
        revisions_root.mkdir(parents=True, exist_ok=True)
        current_path = Path(spec.get("flora_review_path", ""))
        current_revision = int(spec.get("flora_review_revision") or 0)
        if current_path.is_file() and current_revision == 0:
            previous = json.loads(current_path.read_text(encoding="utf-8"))
            current_revision = int(previous.get("revision") or 0)
            if current_revision == 0:
                current_revision = 1
                previous["revision"] = current_revision
                previous["revision_id"] = f"flora-{job_id}-r{current_revision:04d}"
            preserved = revisions_root / f"r{current_revision:04d}" / "annotations.json"
            preserved.parent.mkdir(parents=True, exist_ok=True)
            if not preserved.exists():
                preserved.write_text(json.dumps(previous, ensure_ascii=False, indent=2), encoding="utf-8")
        highest = max((int(path.name[1:]) for path in revisions_root.glob("r[0-9]*") if path.is_dir()), default=0)
        revision = max(current_revision, highest) + 1
        review_path = revisions_root / f"r{revision:04d}" / "annotations.json"
        review_path.parent.mkdir(parents=True, exist_ok=True)
        review = {
            "job_id": job_id,
            "revision": revision,
            "revision_id": f"flora-{job_id}-r{revision:04d}",
            "parent_revision": current_revision or None,
            "dataset_version_id": int(spec["flora_dataset_version_id"]),
            "profile": spec.get("flora_profile"),
            "camera": spec.get("flora_camera"),
            "reviewed_at": reviewed_at,
            "notes": payload.notes.strip(),
            "annotations": annotations,
        }
        temp_path = review_path.with_suffix(".json.tmp")
        temp_path.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(review_path)
        latest = review_root / "annotations.json"
        latest_temp = latest.with_suffix(".json.tmp")
        latest_temp.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
        latest_temp.replace(latest)
        spec["flora_review_path"] = str(review_path)
        spec["flora_reviewed_at"] = reviewed_at
        spec["flora_review_revision"] = revision
        job.spec = spec
        return {"job_id": job_id, "reviewed_at": reviewed_at,
                "revision": revision, "revision_id": review["revision_id"],
                "episodes": len(annotations), "segments": sum(len(x["segments"]) for x in annotations)}


def _materialize_flora_dataset(source_root: Path, target_root: Path, review: dict) -> dict:
    """Create a small LeRobot v2.1 dataset whose per-frame task text uses reviewed FLORA segments."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    meta_in = source_root / "meta"
    info = json.loads((meta_in / "info.json").read_text(encoding="utf-8"))
    task_rows = [json.loads(line) for line in (meta_in / "tasks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    task_texts = {int(row["task_index"]): str(row["task"]) for row in task_rows}
    episodes_by_id = {
        int(row["episode_index"]): row
        for row in (json.loads(line) for line in (meta_in / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
    }
    episode_stats = {}
    stats_path = meta_in / "episodes_stats.jsonl"
    if stats_path.is_file():
        episode_stats = {
            int(row["episode_index"]): row
            for row in (json.loads(line) for line in stats_path.read_text(encoding="utf-8").splitlines() if line.strip())
        }

    reviewed = [item for item in review.get("annotations", []) if item.get("segments")]
    if not reviewed:
        raise ValueError("FLORA review has no episodes with captions")
    target_root.mkdir(parents=True, exist_ok=False)
    (target_root / "meta").mkdir()
    (target_root / "data").mkdir()
    (target_root / "videos").mkdir()
    for item in meta_in.iterdir():
        if item.is_file() and item.name not in {
            "info.json", "tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl",
            "source_episode_mapping.json", "removed_episodes.json",
        }:
            shutil.copy2(item, target_root / "meta" / item.name)

    chunks_size = int(info.get("chunks_size") or 1000)
    data_pattern = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    video_pattern = info.get("video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
    camera_keys = [key for key in (info.get("features") or {}) if key.startswith("observation.images.")]
    task_to_index = {text: index for index, text in task_texts.items()}
    next_task_index = max(task_texts, default=-1) + 1
    derived_episodes, derived_stats, source_map = [], [], []
    sidecar = {
        "format": "embodiedflow.flora-task-sidecar.v1",
        "source_dataset_version_id": int(review["dataset_version_id"]),
        "flora_job_id": int(review["job_id"]),
        "profile": review.get("profile"),
        "camera": review.get("camera"),
        "reviewed_at": review.get("reviewed_at"),
        "review_revision": int(review.get("revision") or 1),
        "review_revision_id": review.get("revision_id") or f"flora-{int(review['job_id'])}-r0001",
        "review_snapshot": "meta/flora_review.json",
        "episodes": [],
        "masks": [],
        "relations": [],
    }
    global_index = 0
    total_frames = 0

    for new_index, annotation in enumerate(sorted(reviewed, key=lambda row: int(row["episode"]))):
        source_index = int(annotation["episode"])
        episode_meta = dict(episodes_by_id[source_index])
        episode_chunk = source_index // chunks_size
        source_data = source_root / data_pattern.format(episode_chunk=episode_chunk, episode_index=source_index)
        if not source_data.is_file():
            raise FileNotFoundError(f"Episode parquet missing: {source_data}")
        table = pq.read_table(source_data)
        if "timestamp" not in table.column_names or "task_index" not in table.column_names:
            raise ValueError("LeRobot parquet must include timestamp and task_index")
        timestamps = table["timestamp"].to_pylist()
        old_task_ids = table["task_index"].to_pylist()
        segments = sorted(annotation["segments"], key=lambda row: float(row["start_s"]))
        task_ids, episode_tasks = [], []
        for timestamp, old_task_id in zip(timestamps, old_task_ids):
            original_text = task_texts.get(int(old_task_id), "")
            matches = [seg for seg in segments
                       if float(seg["start_s"]) <= float(timestamp) < float(seg["end_s"])]
            text = str(matches[-1]["caption"]).strip() if matches else original_text
            if text not in task_to_index:
                task_to_index[text] = next_task_index
                task_texts[next_task_index] = text
                next_task_index += 1
            task_ids.append(task_to_index[text])
            if text and text not in episode_tasks:
                episode_tasks.append(text)

        out_table = table
        out_table = out_table.set_column(
            out_table.schema.get_field_index("task_index"), "task_index",
            pa.array(task_ids, type=out_table.schema.field("task_index").type),
        )
        if "episode_index" in out_table.column_names:
            out_table = out_table.set_column(
                out_table.schema.get_field_index("episode_index"), "episode_index",
                pa.array([new_index] * table.num_rows, type=out_table.schema.field("episode_index").type),
            )
        if "index" in out_table.column_names:
            out_table = out_table.set_column(
                out_table.schema.get_field_index("index"), "index",
                pa.array(range(global_index, global_index + table.num_rows), type=out_table.schema.field("index").type),
            )
        new_chunk = new_index // chunks_size
        data_rel = data_pattern.format(episode_chunk=new_chunk, episode_index=new_index)
        data_out = target_root / data_rel
        data_out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(out_table, data_out)
        global_index += table.num_rows
        total_frames += table.num_rows

        for key in camera_keys:
            video_rel = video_pattern.format(
                episode_chunk=episode_chunk,
                video_key=key,
                episode_index=source_index,
            )
            video_source = source_root / video_rel
            if video_source.is_file():
                video_out_rel = video_pattern.format(
                    episode_chunk=new_chunk,
                    video_key=key,
                    episode_index=new_index,
                )
                video_out = target_root / video_out_rel
                video_out.parent.mkdir(parents=True, exist_ok=True)
                video_out.symlink_to(video_source.resolve())

        episode_meta["episode_index"] = new_index
        episode_meta["length"] = table.num_rows
        episode_meta["tasks"] = episode_tasks
        derived_episodes.append(episode_meta)
        stats = episode_stats.get(source_index)
        if stats:
            stats = dict(stats)
            stats["episode_index"] = new_index
            derived_stats.append(stats)
        source_map.append({"episode_index": new_index, "source_episode_index": source_index})
        sidecar["episodes"].append({"episode_index": new_index, "source_episode_index": source_index,
                                    "segments": annotation["segments"]})
        for mask_ref in annotation.get("mask_run_ids", []):
            mask = _resolve_vision_mask_ref(str(mask_ref))
            artifact_name = mask["run_id"]
            if mask["kind"] == "video":
                artifact_name += f"_frame_{int(mask['frame_index']):05d}"
            mask_rel = Path("meta") / "vision_masks" / f"episode_{new_index:06d}" / f"{artifact_name}.png"
            mask_out = target_root / mask_rel
            mask_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(mask["mask_path"], mask_out)
            mask_source = mask.get("source") or {}
            with Image.open(mask["mask_path"]) as mask_image:
                binary = np.asarray(mask_image.convert("L")) > 0
            ys, xs = np.nonzero(binary)
            height, width = binary.shape
            features = {"width": width, "height": height, "area_pixels": int(binary.sum()),
                        "bbox_xyxy": None, "centroid_xy": None, "centroid_normalized": None}
            if xs.size:
                box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
                centroid = [float(xs.mean()), float(ys.mean())]
                features.update({"bbox_xyxy": box, "centroid_xy": centroid,
                                 "centroid_normalized": [centroid[0] / width, centroid[1] / height]})
                if mask["frame_path"].is_file():
                    with Image.open(mask["frame_path"]) as image:
                        image = image.convert("RGB")
                        if image.size == (width, height):
                            feature_dir = Path("meta") / "vision_features" / f"episode_{new_index:06d}"
                            (target_root / feature_dir).mkdir(parents=True, exist_ok=True)
                            roi_rel = feature_dir / f"{artifact_name}_roi.png"
                            masked_rel = feature_dir / f"{artifact_name}_masked.png"
                            image.crop(tuple(box)).save(target_root / roi_rel)
                            alpha = Image.fromarray(binary.astype(np.uint8) * 255, mode="L")
                            Image.composite(image, Image.new("RGB", image.size), alpha).save(target_root / masked_rel)
                            features.update({"roi_path": str(roi_rel), "masked_rgb_path": str(masked_rel)})
                        else:
                            features["image_status"] = "dimension_mismatch"
                else:
                    features["image_status"] = "source_frame_missing"
            stamp = mask.get("timestamp_s")
            alignment = {"method": "nearest_episode_timestamp", "matched": False}
            if stamp is not None and timestamps:
                nearest = min(range(len(timestamps)), key=lambda i: abs(float(timestamps[i]) - float(stamp)))
                delta = abs(float(timestamps[nearest]) - float(stamp))
                tolerance = 1.0 / float(info.get("fps") or 30)
                alignment.update({"nearest_frame_index": nearest, "frame_timestamp_s": float(timestamps[nearest]),
                                  "delta_s": delta, "tolerance_s": tolerance, "matched": delta <= tolerance})
            sidecar["masks"].append({
                "kind": mask["kind"],
                "mask_ref": mask["id"],
                "episode_index": new_index,
                "source_episode_index": source_index,
                "run_id": mask["run_id"],
                "label": (mask.get("metadata") or {}).get("label") or "unknown",
                "frame_index": mask.get("frame_index"),
                "camera": mask_source.get("camera"),
                "timestamp_s": mask.get("timestamp_s"),
                "filename": mask.get("filename"),
                "path": str(mask_rel),
                "features": features,
                "frame_alignment": alignment,
            })

    from itertools import combinations
    grouped = {}
    for entry in sidecar["masks"]:
        alignment = entry["frame_alignment"]
        if (alignment.get("matched") and entry.get("camera")
                and entry["features"].get("centroid_normalized")):
            key = (entry["episode_index"], entry["camera"], alignment["nearest_frame_index"],
                   entry["features"]["width"], entry["features"]["height"])
            grouped.setdefault(key, []).append(entry)
    for (episode, camera, frame_index, _width, _height), entries in grouped.items():
        for first, second in combinations(entries, 2):
            ax, ay = first["features"]["centroid_normalized"]
            bx, by = second["features"]["centroid_normalized"]
            a, b = first["features"]["bbox_xyxy"], second["features"]["bbox_xyxy"]
            intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
            area_a, area_b = (a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1])
            union = area_a + area_b - intersection
            sidecar["relations"].append({
                "episode_index": episode, "camera": camera, "frame_index": frame_index,
                "first_mask": first["mask_ref"], "second_mask": second["mask_ref"],
                "first_label": first["label"], "second_label": second["label"],
                "direction_relative_to_first": True,
                "delta_xy_normalized": [round(bx-ax, 6), round(by-ay, 6)],
                "distance_normalized": round(math.hypot(bx-ax, by-ay), 6),
                "horizontal": "right" if bx > ax else "left" if bx < ax else "aligned",
                "vertical": "below" if by > ay else "above" if by < ay else "aligned",
                "bbox_iou": round(intersection / union, 6) if union else 0.0,
            })

    info["total_episodes"] = len(derived_episodes)
    info["total_frames"] = total_frames
    info["total_tasks"] = len(task_texts)
    if "total_videos" in info:
        info["total_videos"] = len(derived_episodes) * len(camera_keys)
    info["splits"] = {"train": f"0:{len(derived_episodes)}"}
    (target_root / "meta" / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    (target_root / "meta" / "tasks.jsonl").write_text(
        "".join(json.dumps({"task_index": index, "task": text}, ensure_ascii=False) + "\n"
                for index, text in sorted(task_texts.items())), encoding="utf-8")
    for name, rows in (("episodes.jsonl", derived_episodes), ("episodes_stats.jsonl", derived_stats)):
        (target_root / "meta" / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (target_root / "meta" / "source_episode_mapping.json").write_text(
        json.dumps(source_map, ensure_ascii=False, indent=2), encoding="utf-8")
    review_snapshot = json.dumps(review, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    (target_root / "meta" / "flora_review.json").write_text(review_snapshot, encoding="utf-8")
    sidecar["review_sha256"] = hashlib.sha256(review_snapshot.encode("utf-8")).hexdigest()
    (target_root / "meta" / "flora_annotations.json").write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
    (target_root / "meta" / "removed_episodes.json").write_text("[]\n", encoding="utf-8")
    total_bytes = sum(path.stat().st_size for path in target_root.rglob("*") if path.is_file())
    return {"root": target_root, "episode_count": len(derived_episodes),
            "frame_count": total_frames, "task_count": len(task_texts),
            "total_bytes": total_bytes, "source_episodes": [int(x["episode"]) for x in reviewed],
            "output_episodes": list(range(len(derived_episodes)))}


@router.post("/flora/jobs/{job_id}/derive", status_code=201)
def derive_flora_dataset(job_id: int, revision: int | None = None):
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None or job.kind != "vision_flora":
            raise HTTPException(404, "FLORA job not found")
        spec = dict(job.spec or {})
        review_path = Path(spec.get("flora_review_path", ""))
        if revision is not None:
            if revision < 1:
                raise HTTPException(422, "revision must be positive")
            candidate = Path(spec.get("exp_dir", "")) / "reviewed" / "revisions" / f"r{revision:04d}" / "annotations.json"
            if candidate.is_file():
                review_path = candidate
            elif not (revision == 1 and review_path.is_file() and not spec.get("flora_review_revision")):
                raise HTTPException(404, "FLORA review revision not found")
        if not review_path.is_file():
            raise HTTPException(409, "请先保存人工审核结果")
        review = json.loads(review_path.read_text(encoding="utf-8"))
        annotations = review.get("annotations") or []
        selected = sorted({int(x["episode"]) for x in annotations if x.get("segments")})
        segment_count = sum(len(x.get("segments") or []) for x in annotations)
        if not selected or not segment_count:
            raise HTTPException(400, "审核结果没有可用标注片段")
        source = session.get(DatasetVersion, int(spec["flora_dataset_version_id"]))
        if source is None:
            raise HTTPException(404, "源数据版本不存在")
        source_info = Path(source.manifest_uri or "")
        source_root = source_info.parent.parent if source_info.name == "info.json" else source_info
        stamp_id = uuid.uuid4().hex[:8]
        derived_root = settings.data_root / "datasets" / "derived" / "flora" / f"ds{source.id}_job{job_id}_{stamp_id}"
        try:
            materialized = _materialize_flora_dataset(source_root, derived_root, review)
        except Exception as exc:
            raise HTTPException(500, f"FLORA 数据集生成失败：{type(exc).__name__}: {exc}") from exc
        stamp = int(utcnow().timestamp())
        derived = DatasetVersion(
            dataset_id=source.dataset_id,
            version=f"{source.version[:14]}-flora{stamp:x}{uuid.uuid4().hex[:4]}",
            status="derived",
            episode_count=materialized["episode_count"],
            total_bytes=materialized["total_bytes"],
            manifest_uri=str(materialized["root"] / "meta" / "info.json"),
            episode_indices=None,
        )
        session.add(derived)
        session.flush()
        steps = {
            "operator": "flora_review_to_lerobot_task_index",
            "job_id": job_id,
            "profile": spec.get("flora_profile"),
            "camera": spec.get("flora_camera"),
            "review_path": str(review_path),
            "reviewed_at": review.get("reviewed_at"),
            "review_revision": int(review.get("revision") or 1),
            "review_revision_id": review.get("revision_id") or f"flora-{job_id}-r0001",
            "notes": review.get("notes", ""),
            "source_episodes": materialized["source_episodes"],
            "output_episodes": materialized["output_episodes"],
            "frame_count": materialized["frame_count"],
            "task_count": materialized["task_count"],
            "segment_count": segment_count,
        }
        session.add(PipelineTask(
            job_id=job_id, input_version_id=source.id, output_version_id=derived.id,
            steps=steps, report_uri=str(review_path), status="completed"))
        return {
            "source_version_id": source.id,
            "dataset_version_id": derived.id,
            "version": derived.version,
            "episodes": materialized["episode_count"],
            "segments": segment_count,
            "dataset_root": str(materialized["root"]),
            "review_path": str(review_path),
            "review_revision": int(review.get("revision") or 1),
            "review_revision_id": review.get("revision_id") or f"flora-{job_id}-r0001",
        }



def _flora_export_text(bundle, prefix, segments):
    import csv
    import io
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["start_s", "end_s", "caption"])
    subtitles = []
    def stamp(value):
        milliseconds = max(0, round(float(value) * 1000))
        seconds, ms = divmod(milliseconds, 1000)
        minutes, sec = divmod(seconds, 60)
        hours, minute = divmod(minutes, 60)
        return f"{hours:02}:{minute:02}:{sec:02},{ms:03}"
    for index, segment in enumerate(segments, 1):
        start, end = segment["start_s"], segment["end_s"]
        caption = str(segment.get("caption", ""))
        writer.writerow([start, end, caption])
        subtitles.append(f"{index}\n{stamp(start)} --> {stamp(end)}\n{caption}\n")
    bundle.writestr(prefix + ".csv", "\ufeff" + stream.getvalue())
    bundle.writestr(prefix + ".srt", "\n".join(subtitles))


@router.get("/flora/jobs/{job_id}/download")
def flora_job_download(job_id: int):
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None or job.kind != "vision_flora":
            raise HTTPException(404, "FLORA job not found")
        if job.status != "completed":
            raise HTTPException(409, f"FLORA job status is {job.status}")
        spec = dict(job.spec or {})
    output_dir = Path(spec.get("flora_output_dir", ""))
    run_dir = Path(spec.get("exp_dir", ""))
    archive = run_dir / "flora_annotations.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("manifest.json", json.dumps({
            "dataset_version_id": spec.get("flora_dataset_version_id"),
            "profile": spec.get("flora_profile"),
            "camera": spec.get("flora_camera"),
            "episodes": spec.get("flora_episode_ids", []),
        }, ensure_ascii=False, indent=2))
        for result_id in spec.get("flora_result_ids", []):
            path = output_dir / f"{result_id}.json"
            if not path.is_file():
                raise HTTPException(404, f"FLORA result {result_id} is missing")
            bundle.write(path, arcname=path.name)
            _flora_export_text(bundle, path.stem, json.loads(path.read_text(encoding="utf-8")).get("segments", []))
        review_path = Path(spec.get("flora_review_path", ""))
        if review_path.is_file():
            bundle.write(review_path, arcname="reviewed/annotations.json")
            review = json.loads(review_path.read_text(encoding="utf-8"))
            for item in review.get("annotations", []):
                _flora_export_text(bundle, f"reviewed/episode_{int(item['episode']):06d}", item.get("segments", []))
                for mask_ref in item.get("mask_run_ids", []):
                    mask = _resolve_vision_mask_ref(str(mask_ref))
                    artifact_name = mask["run_id"]
                    if mask["kind"] == "video":
                        artifact_name += f"_frame_{int(mask['frame_index']):05d}"
                    arc_root = f"reviewed/masks/episode_{int(item['episode']):06d}/{artifact_name}"
                    bundle.write(mask["mask_path"], arcname=f"{arc_root}/mask.png")
                    if mask["metadata_path"].is_file():
                        bundle.write(mask["metadata_path"], arcname=f"{arc_root}/metadata.json")
                    if mask["frame_path"].is_file():
                        bundle.write(mask["frame_path"], arcname=f"{arc_root}/frame.jpg")
                    if mask["overlay_path"] and mask["overlay_path"].is_file():
                        bundle.write(mask["overlay_path"], arcname=f"{arc_root}/overlay.png")
            revisions = sorted((run_dir / "reviewed" / "revisions").glob("r[0-9]*/annotations.json"))
            for revision_path in revisions:
                revision = json.loads(revision_path.read_text(encoding="utf-8"))
                revision_prefix = f"reviewed/revisions/{revision_path.parent.name}"
                bundle.write(revision_path, arcname=f"{revision_prefix}/annotations.json")
                for item in revision.get("annotations", []):
                    _flora_export_text(bundle, f"{revision_prefix}/episode_{int(item['episode']):06d}", item.get("segments", []))
    return FileResponse(archive, media_type="application/zip", filename=f"flora_job_{job_id}.zip")

# ---- V08 观测画像（训练与推理共用）----
@router.get("/profiles")
def list_profiles():
    from services.observation_profile import DEFAULT_PROFILE, PROFILES
    return {"default": DEFAULT_PROFILE,
            "profiles": [{"name": p.name, "version": p.version,
                          "dim": len(p.fields), "keys": p.keys()}
                         for p in PROFILES.values()]}


@router.post("/profile/vector")
def profile_vector(payload: dict):
    """把一份派生 Sidecar 转成固定顺序的观测向量。"""
    from services.observation_profile import get_profile, vectorize
    name = payload.get("profile")
    sidecar = payload.get("sidecar") or {}
    try:
        prof = get_profile(name)
    except KeyError as exc:
        raise HTTPException(400, str(exc)) from exc
    return vectorize(prof, sidecar)

@router.get("/tracks/{run_id}/pts.csv")
def track_pts_csv(run_id: str):
    """导出逐帧 PTS 轨迹（含遮挡/丢失状态），可直接给时间轴编辑用。"""
    import csv as _csv
    import io as _io
    import json as _json

    out_dir = _track_dir(run_id)
    meta_p = out_dir / "metadata.json"
    if not meta_p.exists():
        raise HTTPException(404, "跟踪记录不存在")
    meta = _json.loads(meta_p.read_text(encoding="utf-8"))
    fps = float(meta.get("fps") or 0) or None
    frames = meta.get("frames") or []

    buf = _io.StringIO()
    w = _csv.writer(buf)
    w.writerow(["frame_index", "timestamp_s", "pts_s", "status",
                "visibility_status", "visibility_reason",
                "foreground_pixels", "review_suggested", "review_reason"])
    for f in frames:
        idx = f.get("index")
        ts = f.get("timestamp_s")
        w.writerow([idx, ts, ts, f.get("status"),
                    f.get("visibility_status"), f.get("visibility_reason"),
                    f.get("foreground_pixels"),
                    f.get("review_suggested"), f.get("review_reason")])
    body = buf.getvalue()
    return Response(content=body, media_type="text/csv",
                    headers={"content-disposition":
                             f'attachment; filename="{run_id}-pts.csv"'})

@router.get("/assets")
def vision_assets():
    """视觉资产目录：权重路径、体积、下载时间（与版本分开记录）。"""
    import datetime as _dt
    import hashlib as _hash

    def describe(name: str, kind: str, path: Path) -> dict:
        row = {"name": name, "kind": kind, "path": str(path),
               "exists": path.exists(), "size_mb": None,
               "downloaded_at": None, "sha256_12": None, "files": None}
        if not path.exists():
            return row
        st = path.stat()
        row["downloaded_at"] = _dt.datetime.fromtimestamp(
            st.st_mtime, _dt.timezone.utc).isoformat(timespec="seconds")
        if path.is_file():
            row["size_mb"] = round(st.st_size / 1024 / 1024, 1)
            h = _hash.sha256()
            with path.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    h.update(chunk)
            row["sha256_12"] = h.hexdigest()[:12]
        else:
            files = [f for f in path.rglob("*") if f.is_file()]
            row["files"] = len(files)
            row["size_mb"] = round(sum(f.stat().st_size for f in files) / 1024 / 1024, 1)
        return row

    items = [
        describe("sam2.1_hiera_small", "segmenter", CHECKPOINT),
        describe("florence-2-large", "grounding", FLORENCE_DIR),
        describe("grounding-dino-tiny", "grounding", DINO_DIR),
    ]
    total = sum(i["size_mb"] or 0 for i in items)
    return {"root": str(CHECKPOINT.parent), "count": len(items),
            "total_mb": round(total, 1), "assets": items}

@router.get("/ground-batches/{batch_id}/quality")
def batch_quality_gate(batch_id: str, min_score: float = 0.3,
                       min_area: float = 0.0, max_area: float = 1e12):
    """批量标注质量门槛：按置信度与框面积（单位：像素²）过滤，统计通过/拒绝。

    不改变已有产物，只对 results.json 做只读评估。
    """
    import json as _json

    try:
        safe = str(uuid.UUID(batch_id))
    except ValueError as exc:
        raise HTTPException(404, "batch not found") from exc
    rp = VISION_DIR / "ground-batches" / safe / "results.json"
    if not rp.exists():
        raise HTTPException(404, "该批次没有结果文件")
    data = _json.loads(rp.read_text(encoding="utf-8"))

    def boxes_of(node):
        """从任意层级里找出带 score/box 的条目。"""
        out = []
        if isinstance(node, dict):
            if "box" in node or "bbox" in node:
                out.append(node)
            for v in node.values():
                if isinstance(v, (list, dict)):
                    out.extend(boxes_of(v))
        elif isinstance(node, list):
            for v in node:
                out.extend(boxes_of(v))
        return out

    # 真实检测结果在 results.json 的顶层 detections（episodes 只是索引列表）
    items = data.get("detections") or []
    if not items:
        items = boxes_of(data.get("frames"))
    passed, rejected = [], []
    for it in items:
        try:
            score = float(it.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        box = it.get("box") or it.get("bbox") or {}
        try:
            if isinstance(box, dict):
                x1, y1, x2, y2 = (float(box.get(k, 0)) for k in ("x1", "y1", "x2", "y2"))
            else:
                x1, y1, x2, y2 = [float(v) for v in list(box)[:4]]
            area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        except (TypeError, ValueError, IndexError):
            area = 0.0
        rec = {"score": round(score, 4), "area": round(area, 1),
               "label": it.get("label") or it.get("phrase")}
        if score >= min_score and min_area <= area <= max_area:
            passed.append(rec)
        else:
            rejected.append(rec)

    return {
        "batch_id": safe,
        "thresholds": {"min_score": min_score, "min_area": min_area,
                       "max_area": max_area},
        "total": len(items),
        "passed": len(passed),
        "rejected": len(rejected),
        "pass_rate": round(len(passed) / len(items), 4) if items else None,
        "rejected_sample": rejected[:10],
    }

@router.get("/hardware")
def hardware_profile():
    """本机硬件画像：视觉 benchmark 出分时随结果记录，便于跨机器对比。"""
    import platform
    import shutil as _shutil
    import subprocess as _sp

    hw = {"host": platform.node(), "platform": platform.platform(),
          "gpus": [], "cpu": {}, "memory_mb": None}

    # GPU：优先 nvidia-smi，避免为查硬件去加载 torch
    if _shutil.which("nvidia-smi"):
        try:
            out = _sp.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10).stdout
            for line in out.strip().splitlines():
                parts = [x.strip() for x in line.split(",")]
                if len(parts) >= 4:
                    hw["gpus"].append({"index": int(parts[0]), "name": parts[1],
                                       "memory_mb": int(float(parts[2])),
                                       "driver": parts[3]})
        except Exception as exc:
            hw["gpu_error"] = type(exc).__name__

    # CPU 型号
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if line.startswith("model name"):
                    hw["cpu"]["model"] = line.split(":", 1)[1].strip()
                    break
        hw["cpu"]["cores"] = __import__("os").cpu_count()
    except Exception:
        pass

    # 内存
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    hw["memory_mb"] = int(line.split()[1]) // 1024
                    break
    except Exception:
        pass
    return hw

@router.get("/benchmarks/compare")
def compare_vision_benchmarks(dataset_version_id: int | None = None,
                              annotation_run_id: int | None = None):
    """多模型对比：同一 GT 集上各次视觉评估的关键指标并列。

    数据源为 evaluations 表中 source=vision 且 metrics 含 bbox protocol 的记录。
    """
    import json as _json

    from sqlalchemy import select

    from persistence.models import Evaluation

    rows = []
    with session_scope() as s:
        for ev in s.scalars(select(Evaluation).where(Evaluation.source == "vision")
                            .order_by(Evaluation.id.desc())).all():
            raw = ev.metrics
            if isinstance(raw, dict):
                m = raw                      # JSON 列可能已反序列化为 dict
            else:
                try:
                    m = _json.loads(raw or "{}")
                except Exception:
                    continue
            if "protocol" not in m or "bbox" not in str(m.get("protocol", "")):
                continue
            if dataset_version_id is not None and int(m.get("dataset_version_id") or 0) != dataset_version_id:
                continue
            if annotation_run_id is not None and int(m.get("annotation_run_id") or 0) != annotation_run_id:
                continue
            rows.append({
                "evaluation_id": ev.id,
                "model": m.get("model"),
                "mask_model": m.get("mask_model"),
                "prompt": m.get("prompt"),
                "iou_threshold": m.get("iou_threshold"),
                "precision": m.get("precision"),
                "recall": m.get("recall"),
                "f1": m.get("f1"),
                "map50": m.get("map50"),
                "mean_matched_iou": m.get("mean_matched_iou"),
                "tp": m.get("tp"), "fp": m.get("fp"), "fn": m.get("fn"),
                "frames_annotated": m.get("frames_annotated"),
                "frame_coverage": m.get("frame_coverage"),
                "batch_id": m.get("batch_id"),
                "report_uri": ev.report_uri,
            })

    best = None
    for r in rows:
        score = r.get("map50")
        if isinstance(score, (int, float)) and (best is None or score > best):
            best = score
    for r in rows:
        r["is_best_map50"] = (r.get("map50") == best) if best is not None else False

    return {"count": len(rows), "best_map50": best,
            "filters": {"dataset_version_id": dataset_version_id,
                        "annotation_run_id": annotation_run_id},
            "rows": rows}

class MaskIoUIn(BaseModel):
    gt: dict
    pred: dict
    sample: bool = False


@router.post("/mask-iou")
def mask_iou_eval(payload: MaskIoUIn):
    """两份 RLE Mask 的形状评测（IoU/precision/recall）。

    sample=true 表示用的是示例数据，仅验证链路，不能当作模型质量。
    """
    from services.mask_eval import mask_iou, rle_decode
    gt = rle_decode(payload.gt)
    pred = rle_decode(payload.pred)
    if not gt or not pred:
        raise HTTPException(400, "gt/pred 的 RLE 为空或 size=0")
    out = mask_iou(gt, pred)
    out["gt_pixels"] = sum(gt)
    out["pred_pixels"] = sum(pred)
    out["sample"] = payload.sample
    if payload.sample:
        out["note"] = "示例数据，仅验证计算链路，不代表真实模型质量"
    return out


@router.get("/mask-iou/selftest")
def mask_iou_selftest():
    """算法自检：用可手算的矩形用例验证 IoU 与 RLE 实现。"""
    from services.mask_eval import selftest
    return selftest()

class MaskAnnotationIn(BaseModel):
    dataset_version_id: int
    episode_index: int = 0
    camera: str = "front"
    frame_index: int = 0
    label: str | None = None
    rle: dict
    reviewer: str | None = None


def _mask_run(s, dataset_version_id: int, create: bool):
    from sqlalchemy import select
    from persistence.models import AnnotationRun
    run = s.scalars(select(AnnotationRun).where(
        AnnotationRun.kind == "mask_gt",
        AnnotationRun.dataset_version_id == dataset_version_id)).first()
    if run is None and create:
        run = AnnotationRun(dataset_version_id=dataset_version_id, kind="mask_gt",
                            status="ready", episode_count=0, annotation_count=0)
        s.add(run)
        s.flush()
    return run


@router.post("/mask-annotations", status_code=201)
def mask_annotations_create(payload: MaskAnnotationIn):
    """保存一条 mask 标注（RLE），归入该数据版本的 mask_gt 标注集。"""
    from persistence.models import Annotation

    if not payload.rle or not payload.rle.get("counts"):
        raise HTTPException(400, "rle.counts 不能为空")
    with session_scope() as s:
        run = _mask_run(s, payload.dataset_version_id, create=True)
        row = Annotation(
            run_id=run.id, dataset_version_id=payload.dataset_version_id,
            episode_index=payload.episode_index, kind="mask_gt",
            payload={"camera": payload.camera, "frame_index": payload.frame_index,
                     "label": payload.label, "rle": payload.rle},
            confidence=None, status="ready", reviewer=payload.reviewer)
        s.add(row)
        s.flush()
        run.annotation_count = (run.annotation_count or 0) + 1
        return {"id": row.id, "run_id": run.id, "status": "created"}


@router.get("/mask-annotations/{run_id}")
def mask_annotations_list(run_id: int, limit: int = 200):
    from sqlalchemy import select
    from persistence.models import Annotation
    with session_scope() as s:
        rows = s.scalars(select(Annotation).where(Annotation.run_id == run_id)
                         .limit(max(1, min(limit, 1000)))).all()
        items = [{"id": r.id, "episode": r.episode_index,
                  "camera": (r.payload or {}).get("camera"),
                  "frame_index": (r.payload or {}).get("frame_index"),
                  "label": (r.payload or {}).get("label"),
                  "size": ((r.payload or {}).get("rle") or {}).get("size")} for r in rows]
        return {"run_id": run_id, "count": len(items), "items": items}


@router.get("/mask-annotations/{run_id}/export")
def mask_annotations_export(run_id: int, format: str = "json"):
    """导出 mask 标注集：JSON（含 RLE），可直接用于 Mask IoU 评测。"""
    from sqlalchemy import select
    from persistence.models import Annotation, AnnotationRun
    with session_scope() as s:
        run = s.get(AnnotationRun, run_id)
        if run is None or run.kind != "mask_gt":
            raise HTTPException(404, "mask 标注集不存在")
        rows = s.scalars(select(Annotation).where(Annotation.run_id == run_id)).all()
        items = [{"episode": r.episode_index, **(r.payload or {})} for r in rows]
        fmt = (format or "json").lower()
        if fmt not in {"json", "coco", "yolo"}:
            raise HTTPException(400, "format 需为 json / coco / yolo")

        if fmt == "json":
            return {"schema": "embodiedflow.mask_gt/1", "run_id": run_id,
                    "dataset_version_id": run.dataset_version_id,
                    "count": len(items), "items": items}

        if fmt == "coco":
            cats = {}
            images, anns = [], []
            for i, it in enumerate(items, start=1):
                lab = it.get("label") or "object"
                cats.setdefault(lab, len(cats) + 1)
                rle = it.get("rle") or {}
                images.append({"id": i, "file_name":
                               f"ep{it.get('episode',0)}_{it.get('camera','front')}_f{it.get('frame_index',0)}.png",
                               "width": 0, "height": 0})
                anns.append({"id": i, "image_id": i, "category_id": cats[lab],
                             "segmentation": rle, "iscrowd": 0})
            return {"info": {"description": "EmbodiedFlow mask_gt export"},
                    "images": images, "annotations": anns,
                    "categories": [{"id": v, "name": k} for k, v in cats.items()]}

        # yolo：由 RLE 还原估算外接框，归一化到占位边长（真实尺寸未知时按 640 归一）
        lines = []
        SIDE = 640.0
        for it in items:
            rle = it.get("rle") or {}
            size = int(rle.get("size") or 0)
            if not size:
                continue
            side = int(round(size ** 0.5)) or 1
            bits = []
            val = 0
            for c in (rle.get("counts") or []):
                bits.extend([val] * int(c)); val = 1 - val
            xs = [i % side for i, b in enumerate(bits) if b]
            ys = [i // side for i, b in enumerate(bits) if b]
            if not xs:
                continue
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            cx = (x0 + x1 + 1) / 2 / SIDE
            cy = (y0 + y1 + 1) / 2 / SIDE
            w = (x1 - x0 + 1) / SIDE
            h = (y1 - y0 + 1) / SIDE
            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        return {"schema": "embodiedflow.yolo/1", "count": len(lines),
                "note": "外接框按 side=ceil(sqrt(size))、归一化 SIDE=640 估算，仅用于占位导出",
                "lines": lines}

class PoseIn(BaseModel):
    translation: list[float] = [0.0, 0.0, 0.0]
    quaternion: list[float] = [0.0, 0.0, 0.0, 1.0]
    format: str = "quaternion"


@router.get("/pose/selftest")
def pose_selftest():
    """6D Pose 数学层自检（四元数/欧拉/复合/基座映射/坐标轴）。"""
    from services.pose6d import selftest
    return selftest()


@router.post("/pose/convert")
def pose_convert(payload: PoseIn):
    """位姿表示转换：四元数 <-> 旋转矩阵；附坐标轴端点与相机->基座映射结果。"""
    from services.pose6d import matrix_to_quat, pose_axes, quat_to_matrix
    q = payload.quaternion
    R = quat_to_matrix(q)
    return {"quaternion": matrix_to_quat(R), "rotation_matrix": R,
            "axes": pose_axes(payload.translation, q)}


class PoseComposeIn(BaseModel):
    a: PoseIn
    b: PoseIn


@router.post("/pose/compose")
def pose_compose_endpoint(payload: PoseComposeIn):
    from services.pose6d import compose
    return compose(payload.a.translation, payload.a.quaternion,
                   payload.b.translation, payload.b.quaternion)


class PoseToBaseIn(BaseModel):
    point_camera: list[float]
    camera_translation: list[float]
    camera_quaternion: list[float] = [0.0, 0.0, 0.0, 1.0]


@router.post("/pose/to-base")
def pose_to_base(payload: PoseToBaseIn):
    """相机系点 -> 机器人基座坐标（给定相机在基座中的位姿）。"""
    from services.pose6d import camera_to_base
    return {"point_base": camera_to_base(payload.point_camera,
                                         payload.camera_translation,
                                         payload.camera_quaternion)}

class TrajIn(BaseModel):
    model: list[list[float]]
    estimate: list[list[float]]
    align: bool = True


@router.get("/slam/selftest")
def slam_selftest():
    from services.slam_eval import selftest
    return selftest()


@router.post("/slam/ate")
def slam_ate(payload: TrajIn):
    """绝对轨迹误差（对齐后 RMSE 与最大值）。"""
    from services.slam_eval import ate
    try:
        return ate(payload.model, payload.estimate, payload.align)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


class RpeIn(TrajIn):
    delta: int = 1


@router.post("/slam/rpe")
def slam_rpe(payload: RpeIn):
    """相对位姿误差（相隔 delta 帧的相对平移误差）。"""
    from services.slam_eval import rpe
    try:
        return rpe(payload.model, payload.estimate, payload.delta)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

class MaskCompareIn(BaseModel):
    a: dict
    b: dict


@router.post("/mask-annotations/compare")
def mask_revision_compare(payload: MaskCompareIn):
    """比较两份 Mask 修订：IoU、新增/消失像素与变化占比。"""
    from services.mask_eval import mask_iou, rle_decode

    A = rle_decode(payload.a)
    B = rle_decode(payload.b)
    if not A or not B:
        raise HTTPException(400, "a/b 的 RLE 为空")
    n = max(len(A), len(B))
    A = list(A) + [0] * (n - len(A))
    B = list(B) + [0] * (n - len(B))
    added = sum(1 for x, y in zip(A, B) if not x and y)
    removed = sum(1 for x, y in zip(A, B) if x and not y)
    same = sum(1 for x, y in zip(A, B) if x == y)
    out = mask_iou(A, B)
    out.update({"pixels_a": sum(A), "pixels_b": sum(B),
                "added": added, "removed": removed,
                "changed": added + removed,
                "changed_ratio": round((added + removed) / n, 6) if n else 0.0,
                "unchanged_ratio": round(same / n, 6) if n else 0.0})
    return out
