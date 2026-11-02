from __future__ import annotations

import re
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
            free, _total = torch.cuda.mem_get_info(index)
            candidates.append((free, index))
        except Exception:
            pass
    index = max(candidates)[1] if candidates else 0
    torch.cuda.set_device(index)
    return f"cuda:{index}", torch.float16


def load_florence(model_dir: str | Path) -> dict:
    device, dtype = _device_and_dtype()
    processor = AutoProcessor.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True, torch_dtype=dtype
    ).to(device).eval()
    return {"model": model, "processor": processor, "device": device, "dtype": dtype}


def prompt_categories(prompt: str) -> list[str]:
    return list(dict.fromkeys(x.strip() for x in re.split(r"[,，;；\n]+", prompt) if x.strip()))[:12]


def predict_boxes(image: Image.Image, prompt: str, runtime: dict) -> list[dict]:
    model = runtime["model"]
    processor = runtime["processor"]
    device = runtime["device"]
    dtype = runtime["dtype"]
    detections_out = []
    for category in prompt_categories(prompt):
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
        text, spans = processor.post_processor.decode_with_spans(
            processor.tokenizer, sequence.tolist()[1:]
        )
        parsed = processor.post_processor.parse_description_with_bboxes_from_text_and_spans(
            text, spans=spans,
            scores=transition_scores[0].detach().cpu().tolist(),
            score_mode="avg_loc_scores", image_size=image.size,
        )
        for item in parsed:
            raw_box = item.get("bbox") or []
            if len(raw_box) != 4:
                continue
            x1, y1, x2, y2 = (float(value) for value in raw_box)
            x1, x2 = max(0.0, min(x1, image.width)), max(0.0, min(x2, image.width))
            y1, y2 = max(0.0, min(y1, image.height)), max(0.0, min(y2, image.height))
            if x2 <= x1 or y2 <= y1:
                continue
            detections_out.append({
                "label": str(item.get("cat_name") or category),
                "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "score": float(item["score"]) if item.get("score") is not None else None,
            })
    return sorted(detections_out, key=lambda item: item.get("score") or 0.0, reverse=True)
