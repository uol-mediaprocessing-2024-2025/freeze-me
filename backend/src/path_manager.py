from pathlib import Path
import os

FOLDER_PATH = "videos"
IMAGE_PATH = "images"
MASK_PREVIEW_NAME = "mask.png"
MASK_PREVIEW_FOLDER = "mask_previews"
MASKED_VIDEO_NAME = "masked.mp4"
CUT_FOREGROUND_IMAGES_FOLDER = "temp_foreground"
CUT_BACKGROUND_IMAGES_FOLDER = "temp_background"
BACKGROUND_UPLOAD = "background.jpeg"
MOTION_BLUR_FOLDER = "blurred-frames"
TEMP_FILE_NAME = "temp.mp4"
CHECKPOINT_PATH = "checkpoints"
CONFIG_PATH = "configs"

def get_config_path():
    path = os.environ.get("SAM_VERSION").lower()
    if path is None:
        print("SAM_VERSION environment variable is not set")
        print("Possible values are: large, b_plus, small, tiny")
        print("Defaulting to small")

    match path:
        case "large":
            path = Path(CONFIG_PATH).joinpath("sam2.1_hiera_l.yaml").absolute()
        case "b_plus":
            path = Path(CONFIG_PATH).joinpath("sam2.1_hiera_b+.yaml").absolute()
        case "tiny":
            path = Path(CONFIG_PATH).joinpath("sam2.1_hiera_t.yaml").absolute()
        case _:
            path = Path(CONFIG_PATH).joinpath("sam2.1_hiera_s.yaml").absolute()
    return path.as_posix()

def get_checkpoint_path():
    path = os.environ.get("SAM_VERSION").lower()
    if path is None:
        print("SAM_VERSION environment variable is not set")
        print("Possible values are: large, b_plus, small, tiny")
        print("Defaulting to small")

    match path:
        case "large":
            path = Path(CHECKPOINT_PATH).joinpath("sam2.1_hiera_large.pt").absolute()
        case "b_plus":
            path = Path(CHECKPOINT_PATH).joinpath("sam2.1_hiera_base_plus.pt").absolute()
        case "tiny":
            path = Path(CHECKPOINT_PATH).joinpath("sam2.1_hiera_tiny.pt").absolute()
        case _:
            path = Path(CHECKPOINT_PATH).joinpath("sam2.1_hiera_small.pt").absolute()
    return path.as_posix()

def get_video_folder_path(video_id):
    return Path.cwd().joinpath(FOLDER_PATH).joinpath(Path(video_id).stem)

def get_upload_path(video_id):
    return get_video_folder_path(video_id).joinpath(video_id)

def get_background_image(video_id, image_name):
    return get_video_folder_path(video_id).joinpath(image_name)

def get_motion_blur_folder(video_id):
    return get_video_folder_path(video_id).joinpath(MOTION_BLUR_FOLDER)

def get_motion_blur_image(video_id, image_name):
    return get_video_folder_path(video_id).joinpath(image_name)

def get_multiple_instances_image(video_id, image_name):
    return get_video_folder_path(video_id).joinpath(image_name)

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

def get_masked_video_path(video_id):
    return get_video_folder_path(video_id).joinpath(MASKED_VIDEO_NAME)

def get_foreground_temp_image_folder(video_id):
    return get_video_folder_path(video_id).joinpath(CUT_FOREGROUND_IMAGES_FOLDER)

def get_foreground_temp_image_path(video_id, frame_id):
    return get_foreground_temp_image_folder(video_id).joinpath(str(frame_id).zfill(5) + ".png")

def get_background_temp_image_folder(video_id):
    return get_video_folder_path(video_id).joinpath(CUT_BACKGROUND_IMAGES_FOLDER)

def get_background_temp_image_path(video_id, frame_id):
    return get_background_temp_image_folder(video_id).joinpath(str(frame_id).zfill(5) + ".png")

def create_all_paths(video_id):
    Path.cwd().joinpath(FOLDER_PATH).mkdir(parents=True, exist_ok=True)
    get_video_folder_path(video_id).mkdir(parents=True, exist_ok=True)
    get_images_path(video_id).mkdir(parents=True, exist_ok=True)
    get_foreground_temp_image_folder(video_id).mkdir(parents=True, exist_ok=True)
    get_background_temp_image_folder(video_id).mkdir(parents=True, exist_ok=True)
    get_preview_mask_frames_folder_path(video_id).mkdir(parents=True, exist_ok=True)
    get_motion_blur_folder(video_id).mkdir(parents=True, exist_ok=True)