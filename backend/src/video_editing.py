import io
import os
import traceback
from pathlib import Path

import cv2
import supervision as sv
from sam2.sam2_video_predictor import SAM2VideoPredictor
from torch.cuda.amp import autocast
from path_manager import create_all_paths

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import torchvision
import uuid
import numpy as np
from sam2.build_sam import build_sam2_video_predictor

from path_manager import get_video_folder_path
from path_manager import get_background_temp_image_folder
from path_manager import get_foreground_temp_image_folder
from path_manager import get_images_path
from path_manager import get_upload_path
from path_manager import get_checkpoint_path
from path_manager import get_config_path
from path_manager import get_temp_file_path
from path_manager import get_frame_path
from path_manager import get_masked_video_path
from path_manager import get_preview_mask_frames_folder_path
from path_manager import get_preview_mask_frame_name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
print(f"Using checkpoint: {get_checkpoint_path()}")
print(f"Using config: {get_config_path()}")
print(f"Using device: {device}")
print("FlashAttention available:", torch.backends.cuda.flash_sdp_enabled())

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

predictor: SAM2VideoPredictor = build_sam2_video_predictor(get_config_path(), get_checkpoint_path(), device=device)

colors = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']
mask_annotator = sv.MaskAnnotator(
    color=sv.ColorPalette.from_hex(colors),
    color_lookup=sv.ColorLookup.TRACK)

import ffmpeg
from fastapi import UploadFile
import shutil

inference_state: {}
points: []
labels: []
fps = ""


async def save_video(file: UploadFile):
    video_id = uuid.uuid4().hex.__str__() + Path(file.filename).suffix
    create_all_paths(video_id)
    path = get_upload_path(video_id)
    video_data = io.BytesIO(await file.read())
    with open(path, "wb") as f:
        f.write(video_data.getbuffer())
    return video_id

async def get_video_details(video_id):
    try:
        path = get_upload_path(video_id)
        image_folder = get_images_path(video_id)
        details = ffmpeg.probe(path.__str__(), cmd="static_ffprobe")
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
    try:
        global inference_state, points, labels
        image_folder = get_images_path(video_id)
        points = []
        labels = []
        path = get_upload_path(video_id)
        ffmpeg.input(path).output(image_folder.__str__() + "/%05d.jpeg", start_number=0,
                                  **{'q:v': '2'}).overwrite_output().run(quiet=True)
        inference_state = predictor.init_state(video_path=image_folder.__str__())
        return get_frame_path(video_id, 0)
    except Exception as e:
        print(e)
        print(e.__traceback__)
        print(traceback.format_exc())


async def add_new_point_to_segmentation(video_id, point_x, point_y, point_type):
    try:
        global inference_state, points, labels
        points.append([point_x, point_y])
        labels.append(point_type)
        output_path = get_preview_mask_frames_folder_path(video_id)
        with sv.ImageSink(target_dir_path=output_path.__str__()) as sink:
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
            frame_path = get_frame_path(video_id, 0)
            frame = cv2.imread(frame_path.__str__())
            frame = mask_annotator.annotate(frame, detections)
            sink.save_image(frame, get_preview_mask_frame_name(video_id, len(points)))
        return get_preview_mask_frame_name(video_id, len(points))
    except Exception as e:
        print(e)
        print(e.__traceback__)
        print(traceback.format_exc())


async def get_masked_video(video_id):
    try:
        output_path = get_masked_video_path(video_id)
        if output_path.exists():
            return output_path.__str__()
        image_path = get_images_path(video_id)
        video_path = get_upload_path(video_id)
        video_info = sv.VideoInfo.from_video_path(video_path.__str__())
        frames_paths = sorted(sv.list_files_with_extensions(directory=image_path.__str__(), extensions=["jpeg"]))
        # run propagation throughout the video and collect the results in a dict
        temp_file = get_temp_file_path(video_id)
        with sv.VideoSink(temp_file.__str__(), video_info=video_info) as sink:
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
                # Create foreground cut frames
                transparent_foreground = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                transparent_foreground[:, :, 3] = masks.astype(np.uint8) * 255
                cut_frame_path = get_foreground_temp_image_folder(video_id).joinpath(Path(os.path.basename(frames_paths[out_frame_idx])).stem + ".png")
                cv2.imwrite(cut_frame_path, transparent_foreground)

                # Create background cut frames
                transparent_background = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                transparent_background[:, :, 3] = ((masks.astype(np.uint8) + 1) % 2) * 255
                cut_frame_path = get_background_temp_image_folder(video_id).joinpath(Path(os.path.basename(frames_paths[out_frame_idx])).stem + ".png")
                cv2.imwrite(cut_frame_path, transparent_background)

                # Create masked frames
                frame = mask_annotator.annotate(frame, detections)
                sink.write_frame(frame)
        (ffmpeg
         .input(temp_file.__str__())
         .output(
            output_path.__str__(),
            vcodec='libx264',
            movflags='faststart',
            an=None
         )
         .overwrite_output().run(quiet=True)
        )
        temp_file.unlink(missing_ok=True)
        print(output_path.__str__())
        return output_path.__str__()
    except Exception as e:
        print(e)
        print(e.__traceback__)
        print(traceback.format_exc())


async def cut_video(video_id: str, start_time: float, end_time: float):
    """
    Schneidet ein Video auf den angegebenen Zeitraum zu und speichert das geschnittene Video unter einer neuen, zufälligen ID.
    :param video_id: Die ID des hochgeladenen Videos
    :param start_time: Startzeit in Sekunden
    :param end_time: Endzeit in Sekunden
    :return: Der Name des geschnittenen Videos
    """
    try:
        # Pfade initialisieren
        video_folder = get_video_folder_path(video_id)
        original_video_path = get_upload_path(video_id)
        temp_video_path = video_folder.joinpath(f"temp_{video_id}")

        # Überprüfen, ob das Verzeichnis existiert
        if not video_folder.exists():
            raise FileNotFoundError(f"Verzeichnis {video_folder.__str__()} existiert nicht.")

        # Überprüfen, ob das Originalvideo existiert
        if not original_video_path.exists():
            raise FileNotFoundError(f"Originalvideo {original_video_path.__str__()} wurde nicht gefunden.")

        # Video laden
        cap = cv2.VideoCapture(str(original_video_path))
        if not cap.isOpened():
            raise Exception(f"Das Video {original_video_path.__str__()} konnte nicht geladen werden.")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        # Video-Writer konfigurieren
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (width, height))

        # Frames lesen und schreiben
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                print(f"Fehler beim Lesen von Frame {frame_idx}. Abbrechen.")
                break
            out.write(frame)

        cap.release()
        out.release()

        # Überprüfen, ob die temporäre Datei erfolgreich erstellt wurde
        if temp_video_path.exists():
            # Speichern unter einer neuen zufälligen ID
            new_video_file_name = await save_cut_video(temp_video_path, video_id)
            temp_video_path.unlink()  # Temporäre Datei löschen
        else:
            raise Exception(f"Das temporäre Video {temp_video_path} wurde nicht erfolgreich erstellt.")

        return new_video_file_name

    except Exception as e:
        print(f"Fehler beim Schneiden des Videos: {e}")
        raise e


async def save_cut_video(file_path: Path, video_id: str):
    """
    Speichert das Video unter einer zufälligen ID.
    :param file_path: Pfad zur temporären Videodatei
    :return: Der Name der gespeicherten Datei
    """
    path = get_upload_path(video_id)

    # Die temporäre Datei unter der neuen ID speichern
    shutil.copy(file_path, path)  # Kopieren statt verschieben, um die temporäre Datei nicht zu verlieren
    return path
