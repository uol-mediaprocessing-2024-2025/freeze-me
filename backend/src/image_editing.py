import io
import os
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from PIL import Image
from fastapi import UploadFile

from path_manager import get_images_path
from path_manager import get_foreground_temp_image_folder
from path_manager import get_background_image
from path_manager import get_motion_blur_folder
from path_manager import get_motion_blur_image

kernel_list = np.zeros((10, 360), dtype='object')
kernel_list[:, :] = None

last_strength = -1
last_transparency = -1
last_frame_skip = -1


async def save_background(file: UploadFile, video_id):
    path = get_background_image(video_id, "temp_background" + Path(file.filename).suffix)
    image_data = io.BytesIO(await file.read())
    with open(path, "wb") as f:
        f.write(image_data.getbuffer())

    # Convert to jpeg
    final_image_path = get_background_image(video_id, "background.png")
    image = cv2.imread(path)
    cv2.imwrite(final_image_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    path.unlink()
    return final_image_path


async def generate_motion_blur_image(video_id, blur_strength, blur_transparency, frame_skip):

    frames_paths = sorted(sv.list_files_with_extensions(directory=get_foreground_temp_image_folder(video_id).__str__(), extensions=["png"]))
    prev_frame = cv2.imread(frames_paths[0].__str__())
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2BGRA)
    base_image = np.zeros_like(prev_frame)
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    image_path = get_motion_blur_image(video_id, "motion_blur.png")
    cv2.imwrite(image_path, base_image)
    final_image = Image.open(image_path)
    background = Image.open(get_motion_blur_image(video_id, "background.png"))
    final_image.paste(background, (0, 0))

    current_frame = 1 + frame_skip
    while True:
        if current_frame == len(frames_paths):
            break

        next_frame = cv2.imread(frames_paths[current_frame].__str__())
        next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
        print("---------------------")
        print(current_frame)
        print("---------------------")

        global kernel_list
        kernel_list = np.zeros((np.floor(20 * blur_strength).astype(int), 360), dtype='object')
        kernel_list[:, :] = None

        current_frame_path = frames_paths[current_frame - 1].__str__()
        to_be_blurred_frame = cv2.imread(current_frame_path, cv2.IMREAD_UNCHANGED)

        blurred_frame = create_blurred_frame(to_be_blurred_frame, mag, ang, blur_strength, blur_transparency)
        blurred_frame[:, :, 3] *= 0.5 * blur_transparency

        blur_path = get_motion_blur_folder(video_id).joinpath(Path(os.path.basename(current_frame_path)).stem + ".png")
        cv2.imwrite(blur_path, blurred_frame)
        blur_image = Image.open(blur_path)
        final_image.paste(blur_image, (0, 0), blur_image)

        prev_frame = next_frame
        current_frame += 1 + frame_skip
    last_frame = Image.open(frames_paths[len(frames_paths) - 1].__str__())
    final_image.paste(last_frame, (0, 0), last_frame)
    final_image.save(image_path.__str__())
    final_image.show()
    return image_path


def create_blurred_frame(frame, magnitude, angle, blur_strength, blur_transparency):
    global kernel_list
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    blurred_frame = np.zeros_like(frame, dtype=np.float32)
    length = np.zeros_like(magnitude)
    length = cv2.normalize(magnitude, length, 1, 20 * blur_strength, cv2.NORM_MINMAX)
    height, width = frame.shape[:2]

    for y in range(height):
        for x in range(width):
            movement = np.floor(length[y][x]).astype(int)
            direction = np.floor(angle[y][x]).astype(int)

            # Create new kernel if kernel didn't exist before
            if kernel_list[movement - 1][direction - 1] is None:
                kernel = create_motion_blur_kernel(movement, direction)
                kernel_list[movement - 1][direction - 1] = kernel
            else:
                kernel = kernel_list[movement - 1][direction - 1]

            kernel_size = kernel.shape[0]
            x1 = max(x - kernel_size // 2, 0)
            x2 = min(x + kernel_size // 2 + 1, width)
            y1 = max(y - kernel_size // 2, 0)
            y2 = min(y + kernel_size // 2 + 1, height)
            roi = frame[y1:y2, x1:x2]

            if roi.shape[0] >= kernel.shape[0] and roi.shape[1] >= kernel.shape[1]:
                blurred_roi = cv2.filter2D(roi, -1, kernel)
                blurred_frame[y, x] = blurred_roi[kernel_size // 2, kernel_size // 2]
            else:
                blurred_frame[y, x] = frame[y, x]
    return blurred_frame


def create_motion_blur_kernel(magnitude, angle):
    kernel_size = magnitude
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2

    angle_rad = np.deg2rad(angle % 360)
    x0 = center
    y0 = center
    x1 = center + int(np.cos(angle_rad) * center)
    y1 = center + int(np.sin(angle_rad) * center)

    cv2.line(kernel, (x0, y0), (x1, y1), 1, thickness=1)
    kernel /= np.sum(kernel)

    return kernel
