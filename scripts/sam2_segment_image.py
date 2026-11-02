from __future__ import annotations

import contextlib
import json
import select
import sys
import time
from pathlib import Path


class SegmentRuntime:
    def __init__(self):
        self.predictor = None
        self.checkpoint = None
        self.loads = 0
        self.florence = None
        self.florence_path = None
        self.florence_loads = 0
        self.dino = None
        self.dino_loads = 0

    def ground_dino(self, request):
        import inspect
        import re
        import torch
        from PIL import Image
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        started = time.monotonic()
        reused = self.dino is not None
        if not reused:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            processor = AutoProcessor.from_pretrained(request["model_dir"], local_files_only=True)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(
                request["model_dir"], local_files_only=True).to(device).eval()
            self.dino = {"processor": processor, "model": model, "device": device}
            self.dino_loads += 1
        processor, model, device = self.dino["processor"], self.dino["model"], self.dino["device"]
        categories = [part.strip().lower().rstrip(".") for part in re.split(r"[,，;；\n]+", request["prompt"]) if part.strip()]
        text = ". ".join(categories) + "."
        image = Image.open(request["image"]).convert("RGB")
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        threshold = float(request.get("threshold", 0.25))
        kwargs = {"text_threshold": float(request.get("text_threshold", 0.25)),
                  "target_sizes": [image.size[::-1]]}
        parameters = inspect.signature(processor.post_process_grounded_object_detection).parameters
        kwargs["threshold" if "threshold" in parameters else "box_threshold"] = threshold
        result = processor.post_process_grounded_object_detection(outputs, inputs.input_ids, **kwargs)[0]
        labels = result.get("text_labels", result.get("labels", []))
        boxes = []
        for box, score, label in zip(result["boxes"].tolist(), result["scores"].tolist(), labels):
            x1,y1,x2,y2 = box
            boxes.append({"label": str(label), "score": float(score),
                          "box": {"x1": max(0.,x1), "y1": max(0.,y1),
                                  "x2": min(float(image.width),x2), "y2": min(float(image.height),y2)}})
        return {"model": "Grounding DINO Tiny", "prompt": request["prompt"],
                "normalized_prompt": text, "threshold": threshold,
                "text_threshold": kwargs["text_threshold"], "boxes": boxes,
                "width": image.width, "height": image.height, "device": device,
                "model_reused": reused, "model_loads": self.dino_loads,
                "elapsed_s": round(time.monotonic()-started,4)}

    def ground(self, request):
        if request.get("detector") == "grounding-dino-tiny":
            return self.ground_dino(request)
        from PIL import Image
        from florence_runtime import load_florence, predict_boxes
        started = time.monotonic()
        model_dir = str(Path(request["model_dir"]).resolve())
        reused = self.florence is not None and self.florence_path == model_dir
        if not reused:
            self.florence = load_florence(model_dir)
            self.florence_path = model_dir
            self.florence_loads += 1
        image = Image.open(request["image"]).convert("RGB")
        threshold = float(request.get("threshold", 0.2))
        detections = predict_boxes(image, request["prompt"], self.florence)
        return {"model": "Florence-2-large", "prompt": request["prompt"], "threshold": threshold,
                "width": image.width, "height": image.height, "device": self.florence["device"],
                "boxes": [row for row in detections if row.get("score") is None or row["score"] >= threshold],
                "elapsed_s": round(time.monotonic()-started, 4), "model_reused": reused,
                "model_loads": self.florence_loads}

    def segment(self, request):
        import numpy as np
        import torch
        from PIL import Image
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        started = time.monotonic()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = str(Path(request["checkpoint"]).resolve())
        reused = self.predictor is not None and self.checkpoint == checkpoint
        if not reused:
            self.predictor = SAM2ImagePredictor(build_sam2(
                "configs/sam2.1/sam2.1_hiera_s.yaml", checkpoint, device=device))
            self.checkpoint = checkpoint
            self.loads += 1
        image = Image.open(request["image"]).convert("RGB")
        self.predictor.set_image(np.asarray(image))
        points = request.get("points", [])
        coords = np.asarray([[p["x"], p["y"]] for p in points], dtype=np.float32) if points else None
        labels = np.asarray([p["label"] for p in points], dtype=np.int32) if points else None
        box = request.get("box")
        box = np.asarray([box["x1"], box["y1"], box["x2"], box["y2"]], dtype=np.float32) if box else None
        mask_input = None
        if request.get("mask_input"):
            mask_input = np.load(request["mask_input"], allow_pickle=False)[None, ...]
        with torch.inference_mode():
            masks, scores, logits = self.predictor.predict(
                point_coords=coords, point_labels=labels, box=box,
                mask_input=mask_input, multimask_output=True)
        best = int(np.argmax(scores))
        mask = masks[best].astype(np.uint8) * 255
        Image.fromarray(mask, mode="L").save(request["mask"])
        np.save(Path(request["mask"]).with_name("logits.npy"), logits[best], allow_pickle=False)
        return {"width": int(image.width), "height": int(image.height),
                "score": float(scores[best]), "device": device,
                "foreground_pixels": int(mask.sum() // 255),
                "model_reused": reused, "model_loads": self.loads,
                "mask_prompt_used": mask_input is not None,
                "elapsed_s": round(time.monotonic() - started, 4)}


def serve(idle_seconds=120):
    runtime = SegmentRuntime()
    while select.select([sys.stdin], [], [], idle_seconds)[0]:
        line = sys.stdin.readline()
        if not line:
            break
        request = {}
        try:
            request = json.loads(line)
            operation = request.get("operation", "segment")
            if operation == "shutdown":
                break
            if operation == "status":
                result = {"loaded": runtime.predictor is not None, "model_loads": runtime.loads}
            else:
                if operation not in {"segment", "ground"}:
                    raise ValueError("unsupported operation")
                with contextlib.redirect_stdout(sys.stderr):
                    result = runtime.ground(request) if operation == "ground" else runtime.segment(request)
            result["python_prefix"] = sys.prefix
            response = {"request_id": request.get("request_id"), "success": True, "result": result}
        except Exception as error:
            response = {"request_id": request.get("request_id"), "success": False,
                        "error": f"{type(error).__name__}: {error}"}
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--serve":
        serve(float(sys.argv[2]) if len(sys.argv) > 2 else 120)
    elif len(sys.argv) == 2:
        request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        with contextlib.redirect_stdout(sys.stderr):
            result = SegmentRuntime().segment(request)
        print(json.dumps(result))
    else:
        raise SystemExit("usage: sam2_segment_image.py request.json | --serve [idle_seconds]")
