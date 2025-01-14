import io
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from timeit import default_timer as timer

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

last_strength = -1
last_transparency = -1
last_frame_skip = -1

max_kernel_size = 60
min_movement = 1
max_movement = 5
angle_range = 10

kernel_list = np.zeros((max_movement, np.floor(180 / angle_range).astype(int)), dtype='object')
kernel_list[:, :] = None

async def save_background(file: UploadFile, video_id):
    path = get_background_image(video_id, "temp_background" + Path(file.filename).suffix)
    image_data = io.BytesIO(await file.read())
    with open(path, "wb") as f:
        f.write(image_data.getbuffer())

    # Convert to png
    final_image_path = get_background_image(video_id, "background.png")
    image = cv2.imread(path)
    cv2.imwrite(final_image_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    path.unlink()
    return final_image_path


async def generate_motion_blur_image(video_id, blur_strength, blur_transparency, frame_skip):

    frames_paths = sorted(sv.list_files_with_extensions(directory=get_foreground_temp_image_folder(video_id).__str__(), extensions=["png"]))
    prev_frame = cv2.imread(frames_paths[0].__str__())
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2BGRA)
    _, prev_center_x, _, _, prev_center_y, _ = get_max_min_center_of_object(prev_frame)
    base_image = np.zeros_like(prev_frame)
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    image_path = get_motion_blur_image(video_id, "motion_blur.png")
    cv2.imwrite(image_path, base_image)
    final_image = Image.open(image_path)
    background = Image.open(get_motion_blur_image(video_id, "background.png"))
    final_image.paste(background, (0, 0))

    # Pre calculate kernels if not yet exists
    start_time = timer()
    global kernel_list
    kernel_list = np.zeros((max_movement, np.floor(180 / angle_range).astype(int)), dtype='object')
    kernel_list[:, :] = None
    for magnitude in range(kernel_list.shape[0]):
        for angle in range(kernel_list.shape[1]):
            kernel_size = magnitude * 5 + blur_strength * 10 + frame_skip * 5
            kernel_size = np.min([kernel_size, max_kernel_size]).astype(int)

            kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
            kernel[kernel_size // 2, :] = 1.0 / kernel_size
            print(magnitude, angle, kernel)

            center = (kernel_size / 2, kernel_size / 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle * angle_range, 1)
            rotated_kernel = cv2.warpAffine(kernel, rotation_matrix, (kernel_size, kernel_size))
            kernel_list[magnitude][angle] = rotated_kernel
    print("Precalculating kernels took: ", timer() - start_time, " s")

    current_frame = 1 + frame_skip
    while True:
        if current_frame >= len(frames_paths):
            break

        print("---------------------")
        next_frame = cv2.imread(frames_paths[current_frame].__str__())
        next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

        start = timer()
        flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)

        print(current_frame)

        current_frame_path = frames_paths[current_frame - 1].__str__()
        to_be_blurred_frame = cv2.imread(current_frame_path, cv2.IMREAD_UNCHANGED)

        min_x, center_x, max_x, min_y, center_y, max_y = get_max_min_center_of_object(to_be_blurred_frame)
        delta_x = center_x - prev_center_x
        delta_y = center_y - prev_center_y
        angle = np.degrees(np.arctan2(delta_y, delta_x))
        magnitude = np.sqrt(delta_x ** 2 + delta_y ** 2) / np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2) * 8
        magnitude = np.min([magnitude, 5])

        length = np.zeros_like(magnitude)
        length = cv2.normalize(magnitude, length, min_movement, max_movement, cv2.NORM_MINMAX)
        ranged_magnitude = length[min_y:max_y, min_x:max_x]
        ranged_angle = ang[min_y:max_y, min_x:max_x]
        print(np.average(ranged_magnitude), np.median(ranged_magnitude), magnitude)
        print(np.average(ranged_angle), np.median(ranged_angle), angle)

        blurred_frame = create_blurred_frame_global_kernel(to_be_blurred_frame, magnitude, angle, min_x, min_y, max_x, max_y)
        blurred_frame[:, :, 3] = (blurred_frame[:, :, 3] * 0.5 * blur_transparency).astype(np.uint8)

        blur_path = get_motion_blur_folder(video_id).joinpath(Path(os.path.basename(current_frame_path)).stem + ".png")
        cv2.imwrite(blur_path, blurred_frame)
        blur_image = Image.open(blur_path)
        final_image.paste(blur_image, (0, 0), blur_image)
        end = timer()
        print("--- %s seconds ---" % (end - start))
        print("---------------------")

        prev_frame = next_frame
        prev_center_x = center_x
        prev_center_y = center_y
        current_frame += 1 + frame_skip
    last_frame = Image.open(frames_paths[len(frames_paths) - 1].__str__())
    final_image.paste(last_frame, (0, 0), last_frame)
    final_image.save(image_path.__str__())
    final_image.show()
    return image_path

def get_max_min_center_of_object(frame):
    alpha_channel = frame[..., 3]
    visible_pixels = np.argwhere(alpha_channel > 0)
    min_y, min_x = np.min(visible_pixels, axis=0)
    max_y, max_x = np.max(visible_pixels, axis=0)
    center_x = (max_x + min_x) / 2
    center_y = (max_y + min_y) / 2
    return min_x, center_x, max_x, min_y, center_y, max_y

def create_blurred_frame_global_kernel(frame, magnitude, angle, min_x, min_y, max_x, max_y):
    global kernel_list

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    movement = np.floor(magnitude).astype(int)
    direction = np.floor((angle % 180) / angle_range).astype(int)
    print("Values: ", magnitude, angle)
    kernel = kernel_list[movement - 1][direction - 1]
    return cv2.filter2D(frame, -1, kernel)

def create_blurred_frame_experimental(frame, magnitude, angle, min_x, min_y, max_x, max_y):
    global kernel_list

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    blurred_frame = np.zeros_like(frame, dtype=np.float32)
    length = np.zeros_like(magnitude)
    length = cv2.normalize(magnitude, length, min_movement, max_movement, cv2.NORM_MINMAX)
    height, width = frame.shape[:2]

    calc_min_x = min_x - max_kernel_size
    calc_max_x = max_x + max_kernel_size
    calc_min_y = min_y - max_kernel_size
    calc_max_y = max_y + max_kernel_size

    def process_pixel(x,y):
        movement = np.floor(length[y][x]).astype(int)
        direction = np.floor((angle[y][x] % 180) / angle_range).astype(int)
        kernel = kernel_list[movement - 1][direction - 1]

        kernel_size = kernel.shape[0]
        x1, x2 = max(0, x - kernel_size // 2), min(width, x + kernel_size // 2 + 1)
        y1, y2 = max(0, y - kernel_size // 2), min(height, y + kernel_size // 2 + 1)
        roi = frame[y1:y2, x1:x2]
        if roi.shape[0] >= kernel.shape[0] and roi.shape[1] >= kernel.shape[1]:
            blurred_roi = cv2.filter2D(roi, -1, kernel)
            return blurred_roi[kernel_size // 2, kernel_size // 2]
        return frame[y, x]
    with ThreadPoolExecutor():
        for y in range(calc_min_y, calc_max_y):
            for x in range(calc_min_x, calc_max_x):
                blurred_frame[y][x] = process_pixel(x, y)

    return blurred_frame