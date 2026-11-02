from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


TASK = "<OPEN_VOCABULARY_DETECTION>"


def _device_and_dtype():
    if not torch.cuda.is_available():
        return "cpu", torch.float32
    candidates = []
    for index in range(torch.cuda.device_count()):
        try:
            free, total = torch.cuda.mem_get_info(index)
            candidates.append((free, index))
        except Exception:
            pass
    index = max(candidates)[1] if candidates else 0
    torch.cuda.set_device(index)
    return f"cuda:{index}", torch.float16


def _categories(prompt: str) -> list[str]:
    return list(dict.fromkeys(x.strip() for x in re.split(r"[,，;；\n]+", prompt) if x.strip()))[:12]


def main(request_path: str) -> None:
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    started = time.monotonic()
    image = Image.open(request["image"]).convert("RGB")
    device, dtype = _device_and_dtype()
    model_dir = request["model_dir"]
    processor = AutoProcessor.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True, torch_dtype=dtype
    ).to(device).eval()

    boxes_out = []
    for category in _categories(request["prompt"]):
        inputs = processor(text=TASK + category, images=image, return_tensors="pt")
        for key, value in list(inputs.items()):
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=1024, num_beams=3,
                output_scores=True, return_dict_in_generate=True,
            )
            transition_scores = model.compute_transition_scores(
                generated.sequences, generated.scores, generated.beam_indices,
                normalize_logits=True,
            )
        sequence = generated.sequences[0]
        decoded_text, spans = processor.post_processor.decode_with_spans(
            processor.tokenizer, sequence.tolist()[1:]
        )
        detections = processor.post_processor.parse_description_with_bboxes_from_text_and_spans(
            decoded_text,
            spans=spans,
            scores=transition_scores[0].detach().cpu().tolist(),
            score_mode="avg_loc_scores",
            image_size=image.size,
        )
        for detection in detections:
            raw_box = detection.get("bbox", [])
            if len(raw_box) != 4:
                continue
            x1, y1, x2, y2 = (float(value) for value in raw_box)
            x1, x2 = max(0.0, min(x1, image.width)), max(0.0, min(x2, image.width))
            y1, y2 = max(0.0, min(y1, image.height)), max(0.0, min(y2, image.height))
            if x2 <= x1 or y2 <= y1:
                continue
            score = float(detection["score"]) if detection.get("score") is not None else None
            if score is not None and score < float(request.get("threshold", 0.2)):
                continue
            boxes_out.append({
                "label": str(detection.get("cat_name") or category),
                "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "score": score,
            })

    result = {
        "model": "Florence-2-large",
        "prompt": request["prompt"],
        "threshold": float(request.get("threshold", 0.2)),
        "width": image.width,
        "height": image.height,
        "device": device,
        "elapsed_s": round(time.monotonic() - started, 3),
        "boxes": boxes_out,
    }
    Path(request["result"]).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: florence_ground.py request.json")
    main(sys.argv[1])
