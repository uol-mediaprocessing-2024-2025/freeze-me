import io
import os
import traceback
from pathlib import Path

import cv2
import supervision as sv
from PIL import Image
from sam2.sam2_video_predictor import SAM2VideoPredictor

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import torchvision
import uuid
import numpy as np
from sam2.build_sam import build_sam2_video_predictor

CHECKPOINT = Path.cwd().joinpath("checkpoints").joinpath("sam2.1_hiera_large.pt").as_posix()
CONFIG = Path.cwd().joinpath("configs").joinpath("sam2.1_hiera_l.yaml").as_posix()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
print(f"Using checkpoint: {CHECKPOINT}")
print(f"Using config: {CONFIG}")
print(f"Using device: {device}")

predictor: SAM2VideoPredictor = build_sam2_video_predictor(CONFIG, CHECKPOINT, device=device)

colors = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']
mask_annotator = sv.MaskAnnotator(
    color=sv.ColorPalette.from_hex(colors),
    color_lookup=sv.ColorLookup.TRACK)

import ffmpeg
from fastapi import UploadFile

FOLDER_PATH = "videos"
IMAGE_FOLDER_NAME = "images"
inference_state: {}
points: []
labels: []
fps = ""


async def save_video(file: UploadFile):
    Path.cwd().joinpath(FOLDER_PATH).mkdir(parents=True, exist_ok=True)
    video_id = uuid.uuid4().hex.__str__()
    Path.cwd().joinpath(FOLDER_PATH).joinpath(video_id).mkdir(parents=True, exist_ok=True)
    file_name = video_id + Path(file.filename).suffix
    path = Path.cwd().joinpath(FOLDER_PATH).joinpath(video_id).joinpath(file_name)
    video_data = io.BytesIO(await file.read())
    with open(path, "wb") as f:
        f.write(video_data.getbuffer())
    return file_name


async def get_video_details(video_id):
    try:
        parent_folder = Path.cwd().joinpath(FOLDER_PATH).joinpath(Path(video_id).stem)
        path = parent_folder.joinpath(video_id)
        image_folder = parent_folder.joinpath(IMAGE_FOLDER_NAME)
        image_folder.mkdir(parents=True, exist_ok=True)
        details = ffmpeg.probe(path.__str__(), cmd="static_ffprobe")
        ffmpeg.input(path).output(image_folder.__str__() + "/%05d.jpeg", start_number=0,
                                  **{'q:v': '2'}).overwrite_output().run(quiet=True)
        global fps
        fps = (len([name for name in os.listdir(image_folder) if
                    os.path.isfile(os.path.join(image_folder, name))]) / float(details["format"]["duration"]))
        print("FPS: ", fps)
        return details
    except Exception as e:
        print(e)
        print(e.__traceback__)
        print(traceback.format_exc())

        return ""


async def get_first_frame(video_id):
    global inference_state, points, labels
    try:
        points = []
        labels = []
        image_path = Path.cwd().joinpath(FOLDER_PATH).joinpath(Path(video_id).stem).joinpath(IMAGE_FOLDER_NAME)
        print(image_path)
        inference_state = predictor.init_state(video_path=image_path.__str__())
        frame_path = image_path.joinpath("00000.jpeg")
        file = open(frame_path, "rb")
        frame_data = io.BytesIO(file.read())
        return frame_data
    except Exception as e:
        print(e)
        print(e.__traceback__)
        print(traceback.format_exc())


async def add_new_point_to_segmentation(video_id, point_x, point_y, point_type):
    try:
        global inference_state, points, labels
        points.append([point_x, point_y])
        labels.append(point_type)
        print(points)
        print(labels)
        output_path = Path.cwd().joinpath(FOLDER_PATH).joinpath(Path(video_id).stem).joinpath("result")
        with sv.ImageSink(target_dir_path=output_path.__str__(), overwrite=True) as sink:
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                points=points,
                labels=labels,
            )
            mask_data = (out_mask_logits > 0.0).cpu().numpy()
            n, x, h, w = mask_data.shape
            masks = mask_data.reshape(n * x, h, w)
            detections = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks=masks),
                mask=masks,
                tracker_id=np.array(out_obj_ids)
            )
            frame_path = Path.cwd().joinpath(FOLDER_PATH).joinpath(Path(video_id).stem).joinpath(
                IMAGE_FOLDER_NAME).joinpath("00000.jpeg")
            frame = cv2.imread(frame_path.__str__())
            frame = mask_annotator.annotate(frame, detections)
            sink.save_image(frame, "result.png")
        return Image.open(output_path.joinpath("result.png").__str__())
    except Exception as e:
        print(e)
        print(e.__traceback__)
        print(traceback.format_exc())


async def get_masked_video(video_id):
    try:
        image_path = Path.cwd().joinpath(FOLDER_PATH).joinpath(Path(video_id).stem).joinpath(IMAGE_FOLDER_NAME)
        video_path = Path.cwd().joinpath(FOLDER_PATH).joinpath(Path(video_id).stem).joinpath(video_id)
        Path.cwd().joinpath(FOLDER_PATH).joinpath("results").mkdir(parents=True, exist_ok=True)
        output_path = Path.cwd().joinpath(FOLDER_PATH).joinpath("results").joinpath(video_id)
        video_info = sv.VideoInfo.from_video_path(video_path.__str__())
        frames_paths = sorted(sv.list_files_with_extensions(directory=image_path.__str__(), extensions=["jpeg"]))
        # run propagation throughout the video and collect the results in a dict
        with sv.VideoSink(output_path.__str__(), video_info=video_info) as sink:
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                frame = cv2.imread(frames_paths[out_frame_idx])
                masks = (out_mask_logits > 0.0).cpu().numpy()
                n, x, h, w = masks.shape
                masks = masks.reshape(n * x, h, w)
                detections = sv.Detections(
                    xyxy=sv.mask_to_xyxy(masks=masks),
                    mask=masks,
                    tracker_id=np.array(out_obj_ids)
                )
                frame = mask_annotator.annotate(frame, detections)
                sink.write_frame(frame)
        print(output_path)
        return output_path
    except Exception as e:
        print(e)
        print(e.__traceback__)
        print(traceback.format_exc())
