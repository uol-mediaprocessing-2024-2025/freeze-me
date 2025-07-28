import io
import os
import traceback
from pathlib import Path

import cv2
import supervision as sv
from sam2.sam2_video_predictor import SAM2VideoPredictor
from timeit import default_timer as timer


from concurrent.futures import ThreadPoolExecutor
from image_editing import read_images
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

from image_editing import write_images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("CuDNN Version:", torch.backends.cudnn.version())
print(torch.__version__)
print(torch.backends.mkl.is_available())
print(f"Using checkpoint: {get_checkpoint_path()}")
print(f"Using config: {get_config_path()}")
print(f"Using device: {device}")
print("FlashAttention available:", torch.backends.cuda.flash_sdp_enabled())

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    print(f"torch.cuda.get_device_properties(0).major: {torch.cuda.get_device_properties(0).major}")
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
fps = 0
points = []
labels = []


async def save_video(file: UploadFile):
    video_id = uuid.uuid4().hex.__str__() + Path(file.filename).suffix
    create_all_paths(video_id)
    temp_path = get_temp_file_path(video_id)
    path = get_upload_path(video_id)
    video_data = io.BytesIO(await file.read())
    with open(temp_path, "wb") as f:
        f.write(video_data.getbuffer())

    print("saved file at: ", temp_path.__str__())

    ffmpeg.input(temp_path).output(
        path.__str__(),
        vcodec='libx264',
        crf=18,
        preset='slow',
        an=None,
        movflags='faststart'
    ).overwrite_output().run(quiet=True)
    print("converted video and saved at: ", path.__str__())
    temp_path.unlink()
    image_folder = get_images_path(video_id)
    ffmpeg.input(path).output(image_folder.__str__() + "/%05d.jpeg", start_number=0,
                              **{'q:v': '2'}).overwrite_output().run(quiet=True)

    return video_id


async def get_video_details(video_id):
    try:
        path = get_upload_path(video_id)
        image_folder = get_images_path(video_id)
        details = ffmpeg.probe(path.__str__(), cmd="static_ffprobe")
        video_stream = None
        for stream in details["streams"]:
            if stream["codec_type"] == "video":
                video_stream = stream
                break
        global fps
        total_frames = len(os.listdir(image_folder))
        details["total_frames"] = total_frames
        fps_string = video_stream["r_frame_rate"]
        slash = fps_string.find("/")
        fps = round(float(fps_string[0:slash]) / float(fps_string[slash + 1:]), 2)
        print("FPS: ", fps)
        print("Total frames: ", total_frames)
        return details
    except Exception as e:
        print(e)
        print(e.__traceback__)
        print(traceback.format_exc())

        return ""


async def initialize_segmentation(video_id):
    try:
        global inference_state, points, labels
        image_folder = get_images_path(video_id)
        total_frames = len(os.listdir(image_folder))
        points = [[[]] for _ in range(total_frames)]
        labels = [[[]] for _ in range(total_frames)]
        print(points)
        print(labels)
        inference_state = predictor.init_state(video_path=image_folder.__str__())
    except Exception as e:
        print(e)
        print(e.__traceback__)
        print(traceback.format_exc())


async def get_frame(video_id, frame_id):
    try:
        return get_frame_path(video_id, frame_id)
    except Exception as e:
        print(e)
        print(e.__traceback__)
        print(traceback.format_exc())


async def add_new_point_to_segmentation(video_id, point_x, point_y, point_type, frame_num, object_num):
    try:
        global points, labels
        if len(points[frame_num]) < object_num:
            points[frame_num].append([])
            labels[frame_num].append([])
        points[frame_num][object_num - 1].append([point_x, point_y])
        labels[frame_num][object_num - 1].append(point_type)
        print(points)
        print(labels)
        output_path = get_preview_mask_frames_folder_path(video_id)
        with sv.ImageSink(target_dir_path=output_path.__str__()) as sink:
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_num,
                obj_id=object_num,
                points=points[frame_num][object_num - 1],
                labels=labels[frame_num][object_num - 1],
            )
            mask_data = (out_mask_logits > 0.0).cpu().numpy()
            n, x, h, w = mask_data.shape
            masks = mask_data.reshape(n * x, h, w)
            detections = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks=masks),
                mask=masks,
                tracker_id=np.array(out_obj_ids)
            )
            frame_path = get_frame_path(video_id, frame_num)
            frame = cv2.imread(frame_path.__str__())
            frame = mask_annotator.annotate(frame, detections)
            sink.save_image(frame, get_preview_mask_frame_name(video_id, frame_num))
        return get_preview_mask_frame_name(video_id, frame_num)
    except Exception as e:
        print(e)
        print(e.__traceback__)
        print(traceback.format_exc())

async def get_masked_video_better(video_id):
    try:
        load_start = timer()
        image_path = get_images_path(video_id)
        frames_paths = sorted(sv.list_files_with_extensions(directory=image_path.__str__(), extensions=["jpeg"]))
        background_paths = [f'{get_background_temp_image_folder(video_id).joinpath(Path(os.path.basename(x)).stem + ".png")}' for x in frames_paths]
        foreground_paths = [f'{get_foreground_temp_image_folder(video_id).joinpath(Path(os.path.basename(x)).stem + ".png")}' for x in frames_paths]
        segmented_paths = [f'{get_preview_mask_frames_folder_path(video_id).joinpath(Path(os.path.basename(x)).stem + ".jpeg")}' for x in frames_paths]

        frames = read_images(frames_paths)
        load_end = timer()
        print(f"Finished loading paths and reading frames.")
        print("--- Time: %s seconds ---" % (load_end - load_start))
        print("---------------------")

        ids = [[] for _ in range(len(frames_paths))]
        masks = [[] for _ in range(len(frames_paths))]
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=0):
            ids[out_frame_idx] = np.array(out_obj_ids)
            masks[out_frame_idx] = (out_mask_logits > 0.0).cpu().numpy()

        segmentation_end = timer()
        print(f"Finished segmentation.")
        print("--- Time: %s seconds ---" % (segmentation_end - load_end))
        print("---------------------")

        def process_frame(frame_index):
            mask = np.array(masks[frame_index])
            n, x, h, w = mask.shape
            out_masks = mask.reshape(n * x, h, w)

            detection = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks=out_masks),
                mask=out_masks,
                tracker_id=np.array(ids[frame_index])
            )
            return mask_annotator.annotate(frames[frame_index], detection)

        with ThreadPoolExecutor() as executor:
            segmented_frames = list(executor.map(process_frame, range(len(frames))))

        detection_end = timer()
        print(f"Finished detections, masks recalculations and segmented image.")
        print("--- Time: %s seconds ---" % (detection_end - segmentation_end))
        print("---------------------")

        def process_alpha(frame_index):
            mask_list = masks[frame_index]
            combined_mask = np.logical_or.reduce(mask_list)

            frame = frames[frame_index]
            transparent_foreground = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            transparent_foreground[:, :, 3] = combined_mask.astype(np.uint8) * 255

            transparent_background = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            transparent_background[:, :, 3] = ((combined_mask.astype(np.uint8) + 1) % 2) * 255

            return transparent_foreground, transparent_background

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_alpha, range(len(frames))))
        foreground_frames, background_frames = zip(*results)

        masked_images_end = timer()
        print(f"Finished creating foreground and background images.")
        print("--- Time: %s seconds ---" % (masked_images_end - detection_end))
        print("---------------------")

        write_images(foreground_paths, foreground_frames)
        write_images(background_paths, background_frames)
        write_images(segmented_paths, segmented_frames)

        saving_end = timer()
        print(f"Finished saving all images.")
        print("--- Time: %s seconds ---" % (saving_end - masked_images_end))
        print("---------------------")


        print("#####################")
        print("--- TOTAL TIME: %s seconds ---" % (saving_end - load_start))
        print("---------------------")


    except Exception as e:
        print(e)
        print(e.__traceback__)
        print(traceback.format_exc())


async def cut_video(video_id: str, start_time: float, end_time: float):
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

        global fps
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        print(f"Start Frame: {start_frame}")
        print(f"End Frame: {end_frame}")
        print(f"FPS: {fps}")

        input_file = ffmpeg.input(original_video_path.__str__())
        ffmpeg.output(input_file.trim(start_frame=start_frame, end_frame=end_frame).setpts('PTS-STARTPTS'),
                      temp_video_path.__str__(), vcodec='libx264', movflags='faststart',
                      an=None).overwrite_output().run(quiet=True)

        # Überprüfen, ob die temporäre Datei erfolgreich erstellt wurde
        if temp_video_path.exists():
            # Speichern unter einer neuen zufälligen ID
            new_video_file_name = await save_cut_video(temp_video_path, video_id)
            temp_video_path.unlink()  # Temporäre Datei löschen
        else:
            raise Exception(f"Das temporäre Video {temp_video_path} wurde nicht erfolgreich erstellt.")
        image_folder = get_images_path(video_id)
        for f in os.listdir(image_folder):
            os.remove(os.path.join(image_folder, f))
        ffmpeg.input(original_video_path).output(image_folder.__str__() + "/%05d.jpeg", start_number=0,
                                                 **{'q:v': '2'}).overwrite_output().run(quiet=True)
        return new_video_file_name

    except Exception as e:
        print(f"Fehler beim Schneiden des Videos: {e}")
        raise e


async def save_cut_video(file_path: Path, video_id: str):
    path = get_upload_path(video_id)

    shutil.copy(file_path, path)
    return path
