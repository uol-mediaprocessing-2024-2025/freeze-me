import io
import os
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import torchvision
from sam2.build_sam import build_sam2_video_predictor

CHECKPOINT = Path.cwd().joinpath("backend").joinpath("src").joinpath("checkpoints").joinpath(
    "sam2.1_hiera_large.pt").as_posix()
CONFIG = Path.cwd().joinpath("backend").joinpath("src").joinpath("configs").joinpath("sam2.1_hiera_l.yaml").as_posix()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
print(f"Using checkpoint: {CHECKPOINT}")
print(f"Using config: {CONFIG}")
print(f"Using device: {device}")

sam2_model = build_sam2_video_predictor(CONFIG, CHECKPOINT, device=device)

import ffmpeg
from fastapi import UploadFile

FOLDER_PATH = "videos"

async def get_video_details(file: UploadFile):
    try:
        Path.cwd().joinpath(FOLDER_PATH).mkdir(parents=True, exist_ok=True)
        path = Path.cwd().joinpath(FOLDER_PATH).joinpath(file.filename)
        video_data = io.BytesIO(await file.read())
        with open(path, "wb") as f:
            f.write(video_data.getbuffer())
        details = ffmpeg.probe(path.__str__(), cmd="static_ffprobe")
        return details
    except Exception as e:
        print(e)
        print(e.__traceback__)
        return ""

async def cut_object_from_video(path):
    return path
