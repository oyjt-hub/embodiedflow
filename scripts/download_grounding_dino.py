import json,traceback
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
out=Path("/home/oyjt/workspace/embodiedflow_data/models/vision/grounding-dino-tiny")
out.mkdir(parents=True,exist_ok=True)
status=out/"download_status.json"
try:
 info=HfApi(endpoint="https://hf-mirror.com").model_info("IDEA-Research/grounding-dino-tiny")
 status.write_text(json.dumps({"state":"downloading","repo":info.id,"revision":info.sha}))
 snapshot_download(info.id,revision=info.sha,local_dir=str(out),endpoint="https://hf-mirror.com",allow_patterns=["*.json","*.txt","*.safetensors"],max_workers=2)
 status.write_text(json.dumps({"state":"complete","repo":info.id,"revision":info.sha}))
 print(status.read_text(),flush=True)
except Exception as e:
 status.write_text(json.dumps({"state":"failed","error":str(e)}));raise
