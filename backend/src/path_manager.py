from pathlib import Path
import os

FOLDER_PATH = "videos"
IMAGE_PATH = "images"
MASK_PREVIEW_NAME = "mask.png"
MASK_PREVIEW_FOLDER = "mask_previews"
RESULT_NAME = "result.mp4"
TEMP_FILE_NAME = "temp.mp4"
CHECKPOINT_PATH = "checkpoints"
CONFIG_PATH = "configs"
WORK_DIRECTORY = Path.cwd()


def get_config_path():
    path = os.environ.get("CONFIG")
    if path is None:
        raise Exception("CONFIG environment variable not set")
    return Path(path).as_posix()

def get_checkpoint_path():
    path = os.environ.get("CHECKPOINT")
    if path is None:
        raise Exception("CHECKPOINT environment variable not set")
    return Path(path).as_posix()

def get_video_folder_path(video_id):
    return Path.cwd().joinpath(FOLDER_PATH).joinpath(Path(video_id).stem)

def get_upload_path(video_id):
    return get_video_folder_path(video_id).joinpath(video_id)

def get_images_path(video_id):
    return get_video_folder_path(video_id).joinpath(IMAGE_PATH)

def get_frame_path(video_id, frame_id):
    return get_images_path(video_id).joinpath(str(frame_id).zfill(5) + ".jpeg")

def get_preview_mask_frames_folder_path(video_id):
    return get_video_folder_path(video_id).joinpath(MASK_PREVIEW_FOLDER)

def get_preview_mask_frame_name(video_id, frame_id):
    return get_preview_mask_frames_folder_path(video_id).joinpath(str(frame_id).zfill(5) + "_" + MASK_PREVIEW_NAME)

def get_temp_file_path(video_id):
    return get_video_folder_path(video_id).joinpath(TEMP_FILE_NAME)

def get_result_path(video_id):
    return get_video_folder_path(video_id).joinpath(RESULT_NAME)

def create_all_paths(video_id):
    Path.cwd().joinpath(FOLDER_PATH).mkdir(parents=True, exist_ok=True)
    Path.cwd().joinpath(FOLDER_PATH).joinpath(Path(video_id).stem).mkdir(parents=True, exist_ok=True)
    Path.cwd().joinpath(FOLDER_PATH).joinpath(Path(video_id).stem).joinpath(IMAGE_PATH).mkdir(parents=True, exist_ok=True)
    Path.cwd().joinpath(FOLDER_PATH).joinpath(Path(video_id).stem).joinpath(MASK_PREVIEW_FOLDER).mkdir(parents=True, exist_ok=True)
