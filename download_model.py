#!/usr/bin/env python3
import os
from pathlib import Path
from huggingface_hub import hf_hub_download
repo=os.getenv('TACTICAI_YOLO_REPO','hugosanchezgallego/futbol-detector-v4')
filename=os.getenv('TACTICAI_YOLO_FILENAME','best.pt')
out=Path(os.getenv('TACTICAI_MODEL_CACHE','./models')); out.mkdir(parents=True,exist_ok=True)
print(hf_hub_download(repo_id=repo,filename=filename,local_dir=str(out),local_dir_use_symlinks=False))