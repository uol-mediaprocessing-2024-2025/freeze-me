import json
import os
from enum import Enum
from pathlib import Path

from path_manager import get_main_folder
from path_manager import get_workflow_data_path

# saved data keys
CURRENT_STEP = "current_step"
ALL_STEPS = "all_steps"
AVAILABLE_STEPS = "available_steps"
ID = "id"

# steps after uploading video
class Step(Enum):
    VIDEO_EDITING = "video-editing"
    SEGMENTATION = "segmentation"
    BACKGROUND = "background"
    MAIN_EFFECT = "main-effect"
    AFTER_EFFECT = "after-effect"


def create_project(video_id):
    workflow_data = {CURRENT_STEP: Step.VIDEO_EDITING.value, AVAILABLE_STEPS: [Step.VIDEO_EDITING.value], ID: video_id}
    print(video_id)
    print(get_workflow_data_path(video_id))
    save_data(get_workflow_data_path(video_id), workflow_data)


def get_all_projects():
    path = get_main_folder()
    video_ids = os.listdir(path.__str__())
    print(video_ids)
    return video_ids


def get_current_step(video_id):
    return load_data(get_workflow_data_path(video_id))[CURRENT_STEP]


def set_current_step(video_id, step: Step):
    workflow_data = load_data(get_workflow_data_path(video_id))
    workflow_data[CURRENT_STEP] = step.value
    if step.value not in workflow_data[AVAILABLE_STEPS]:
        workflow_data[AVAILABLE_STEPS].append(step.value)
    save_data(get_workflow_data_path(video_id), workflow_data)


def get_step_data(video_id):
    return load_data(get_workflow_data_path(video_id))


def load_data(path):
    if Path(path).exists() and Path(path).is_file():
        f = open(path, "r")
        return json.load(f)
    else:
        raise Exception(f"Could not load data from: {path}")


def save_data(path, data):
    if not Path(path).exists():
        Path(path).touch()
    f = open(path, "w")
    f.write(json.dumps(data))
