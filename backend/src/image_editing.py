import functools
import io
import os
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from timeit import default_timer as timer
from scipy.signal import savgol_filter

import cupy
import cupyx
import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image, ImageColor
from cupyx.scipy import ndimage
from fastapi import UploadFile

from path_manager import get_background_image, get_background_temp_image_path
from path_manager import get_background_temp_image_folder
from path_manager import get_foreground_temp_image_folder
from path_manager import get_motion_blur_folder
from path_manager import get_motion_blur_image
from project_data import set_motion_blur_metadata, get_motion_blur_data, get_background_type, BackgroundType

max_kernel_size = 60
min_movement = 1
max_movement = 5
angle_range = 10

kernel_list = np.zeros((max_movement, np.floor(180 / angle_range).astype(int)), dtype='object')
kernel_list[:, :] = None

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


async def save_background(file: UploadFile, video_id):
    path = get_background_image(video_id, "temp_background" + Path(file.filename).suffix)
    image_data = io.BytesIO(await file.read())
    with open(path, "wb") as f:
        f.write(image_data.getbuffer())

    # Convert to rgba-png
    final_image_path = get_background_image(video_id, "custom_background.png")
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    cv2.imwrite(final_image_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    path.unlink()
    return final_image_path


def get_background(video_id, frame_id):
    background_type: BackgroundType = get_background_type(video_id)
    if background_type == BackgroundType.CUSTOM.value:
        background = get_custom_background(video_id)
    elif background_type == BackgroundType.TRANSPARENT.value:
        background = get_transparent_background(video_id, frame_id)
    else:
        background = get_video_frame_background(video_id, frame_id)
    return background


def get_custom_background(video_id):
    background_path = get_background_image(video_id, "custom_background.png")
    background = cv2.imread(background_path, cv2.IMREAD_UNCHANGED)
    return background


def get_video_frame_background(video_id, frame_id):
    background_path = get_background_temp_image_path(video_id, frame_id)
    background = cv2.imread(background_path, cv2.IMREAD_UNCHANGED)
    return background


def get_transparent_background(video_id, frame_id):
    frame_path = get_background_temp_image_path(video_id, frame_id)
    frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
    return np.zeros_like(frame)


async def create_motion_blur_image(video_id, blur_strength, blur_transparency, frame_skip):
    device = get_device()
    motion_blur_data = get_motion_blur_data(video_id)
    if not motion_blur_data:
        # If data doesn't exist, this is the first time generating motion blur image
        generate_blur = True
        motion_blur_data = [0, 0, 0]
    else:
        generate_blur = motion_blur_data[0] != blur_strength or motion_blur_data[1] != blur_transparency or \
                        motion_blur_data[2] != frame_skip
    if device == "cuda":
        path = gpu_motion_blur(video_id, blur_strength, blur_transparency, frame_skip, generate_blur)
    else:
        path = generate_motion_blur_image(video_id, blur_strength, blur_transparency, frame_skip)

    # update motion blur data
    motion_blur_data[0] = blur_strength
    motion_blur_data[1] = blur_transparency
    motion_blur_data[2] = frame_skip
    set_motion_blur_metadata(video_id, motion_blur_data)

    return path


def gpu_motion_blur(video_id, blur_strength, blur_transparency, frame_skip, generate_blur):
    # Get needed frames: regular if new blur is generated and already blurred frames if blur isn't necessary
    result_path = get_motion_blur_image(video_id, "motion_blur.png")
    if not generate_blur:
        return result_path
    if generate_blur:
        frame_path = get_foreground_temp_image_folder(video_id).__str__()
    else:
        frame_path = get_motion_blur_folder(video_id).__str__()
    frames_paths = sorted(sv.list_files_with_extensions(directory=frame_path, extensions=["png"]))
    used_frame_paths = []
    load_start = timer()
    last_frame_id = len(frames_paths) - 1
    for i in range(last_frame_id, -1, -(1 + frame_skip)):
        print(i)
        used_frame_paths.insert(0, frames_paths[i].__str__())
    print(last_frame_id)
    used_frames = read_images(used_frame_paths)
    load_end = timer()
    print("--- Loading: %s seconds ---" % (load_end - load_start))
    print("---------------------")
    print("Loaded: " + str(len(used_frames)) + " frames.")

    # Calculate how many images can be in a batch for the gpu
    mem_info = cupy.cuda.Device(0).mem_info
    free_mem = mem_info[0] / 1024 ** 2
    byte_size = used_frames[0].size
    mb_size = byte_size / 1024 ** 2
    max_images = round(free_mem / mb_size)
    print(f"Free GPU-VRAM: {free_mem:.2f} MB")
    print(f"Maximum amount of MB per image: {mb_size}")
    print(f"Maximum amount of images in VRAM: {max_images}")

    # Set batch size lower than maximum possible just in case
    batch_size = max_images / 2
    batch_size = round(batch_size * 0.75)
    batch_size = len(used_frames) if batch_size > len(used_frames) else batch_size

    resulting_image = get_background(video_id, last_frame_id)
    resulting_image_gpu = cupy.array(resulting_image, dtype=cupy.uint8)

    # generate image batch for batch
    for i in range(0, len(used_frames), batch_size):
        # load frames into gpu
        batch = used_frames[i:i + batch_size]
        gpu_start = timer()
        batch_gpu = [cupy.array(image, dtype=cupy.uint8) for image in batch]
        gpu_end = timer()
        print("--- Transfer to GPU: %s seconds ---" % (gpu_end - gpu_start))
        print("---------------------")

        # generate blur if needed
        start = timer()
        if generate_blur:
            # calculate necessary data
            with ThreadPoolExecutor() as executor:
                streams = [cupy.cuda.Stream() for _ in batch_gpu]
                rois = list(executor.map(get_roi_gpu, batch_gpu, streams))
                print("finished rois")
                centers = list(executor.map(get_center_gpu, rois, streams))
                print("finished centers")
                deltas = list(executor.map(get_delta_gpu, centers[:-1], centers[1:], streams[:-1]))
                print("finished deltas")
                angles = list(executor.map(get_angle_gpu, deltas, streams[:-1]))
                print("finished angles")
                magnitudes = list(executor.map(get_magnitude_gpu, deltas, rois[:-1], streams[:-1]))
                print("finished magnitudes")
                [stream.synchronize() for stream in streams]
                del streams
            prep_time = timer()
            print("--- Preparing: %s seconds ---" % (prep_time - start))
            print("---------------------")

            print("magnitude: ", cupy.array(magnitudes))
            # angles = average_out_angle_gpu(angles)
            with ThreadPoolExecutor() as executor:
                streams = [cupy.cuda.Stream() for _ in batch_gpu[:-1]]
                gpu_blur_image_with_defaults = functools.partial(
                    gpu_blur_image,
                    blur_strength=blur_strength,
                    blur_transparency=blur_transparency,
                    frame_skip=frame_skip,
                    show=False
                )
                blurred_batch_gpu = list(executor.map(gpu_blur_image_with_defaults, batch_gpu[:-1], angles, magnitudes, rois[:-1], centers[:-1], streams))
                [stream.synchronize() for stream in streams]
                del streams
            blur_time = timer()
            print("--- Blurring: %s seconds ---" % (blur_time - prep_time))
            print("---------------------")

            blurred_batch_gpu.append(batch_gpu[-1])
            print("---------------------")
        else:
            blurred_batch_gpu = batch_gpu
        comp_start = timer()
        for j in range(0, len(batch_gpu)):
            resulting_image_gpu = alpha_composite_gpu(blurred_batch_gpu[j], resulting_image_gpu)

        comp_end = timer()
        print("--- Composite: %s seconds ---" % (comp_end - comp_start))
        print("---------------------")
    # save and return
    resulting_image = cupy.asnumpy(resulting_image_gpu)
    cv2.imwrite(result_path.__str__(), resulting_image)
    cupy.get_default_memory_pool().free_all_blocks()
    return result_path


def cpu_motion_blur(video_id, blur_strength, blur_transparency, frame_skip, generate_blur):
    return Path(get_motion_blur_image(video_id, "motion_blur.png"))


def gpu_blur_image(image, angle, magnitude, rois, center, stream, blur_strength, blur_transparency, frame_skip, show):
    with stream:
        # create kernel based on blur_strength, magnitude and frame_skip
        kernel_size = int(magnitude * 15 + blur_strength * 15 + frame_skip * 5)
        kernel = cupy.zeros((1, kernel_size), dtype=cupy.float32)
        kernel[0] = 1.0 / kernel_size

        # Crop ROI
        cropped = image[rois[1]:rois[3], rois[0]:rois[2]]
        center_y = center[1]
        center_x = center[0]

        # Rotate ROI
        rotated = cupyx.scipy.ndimage.rotate(cropped, angle, reshape=True, mode='constant', cval=0)

        # Stretch and blur
        stretched = cupyx.scipy.ndimage.zoom(rotated, (0.9, magnitude, 1), order=1)
        padded = cupy.pad(stretched, pad_width=((0, 0), (kernel_size, kernel_size), (0, 0)), mode='constant', constant_values=0)

        result = cupy.zeros_like(padded)
        for channel in range(4):  # Loop over RGBA channels
            result[:, :, channel] = cupyx.scipy.ndimage.convolve(padded[:, :, channel], kernel, mode='constant', cval=0)

        # Rotate back
        derotated = cupyx.scipy.ndimage.rotate(result, -angle, reshape=True, mode='constant', cval=0)

        # Calculate new x and y coordinates (necessary because of changed crop-size)
        derotated_height, derotated_width = derotated.shape[0], derotated.shape[1]
        new_y1 = int(center_y - derotated_height // 2)
        new_y2 = int(new_y1 + derotated_height)
        new_x1 = int(center_x - derotated_width // 2)
        new_x2 = int(new_x1 + derotated_width)

        crop_y1 = max(0, -new_y1)
        crop_y2 = derotated_height - max(0, new_y2 - image.shape[0])
        crop_x1 = max(0, -new_x1)
        crop_x2 = derotated_width - max(0, new_x2 - image.shape[1])

        derotated_cropped = derotated[crop_y1:crop_y2, crop_x1:crop_x2]

        new_y1 = max(0, new_y1)
        new_y2 = min(image.shape[0], new_y2)
        new_x1 = max(0, new_x1)
        new_x2 = min(image.shape[1], new_x2)

        # Put back in original frame
        old_crop_mask = np.zeros_like(cropped)
        image[rois[1]:rois[3], rois[0]:rois[2]] = old_crop_mask     # deletes unblurred input
        image[new_y1:new_y2, new_x1:new_x2] = derotated_cropped     # adds blurred input

        # Add transparency
        image[:, :, 3] = (image[:, :, 3] * blur_transparency).astype(np.uint8)

    return image


def cpu_blur_image(image, blur_strength, blur_transparency, frame_skip, angle, magnitude, rios):
    # create kernel based on blur_strength, magnitude and frame_skip
    kernel_size = magnitude * 2 + blur_strength * 10 + frame_skip * 5
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2] = 1.0 / kernel_size

    # Crop ROI
    cropped = image[rios[0]:rios[2], rios[1], rios[3]]
    width = rios[2] - rios[0]
    height = rios[3] - rios[1]

    # Rotate ROI
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    rotated = cv2.warpAffine(cropped, rotation_matrix, (width, height), flags=cv2.INTER_CUBIC)

    # Stretch and blur
    stretched = cv2.resize(rotated, None, None, 1.0, 1.0, cv2.INTER_LINEAR)
    result = cv2.filter2D(stretched, -1, kernel)

    # Rotate back
    inverse_matrix = cv2.getRotationMatrix2D(center, -angle, 1)
    unrotated = cv2.warpAffine(result, inverse_matrix, (width, height), flags=cv2.INTER_CUBIC)

    # Put back in original frame
    image[rios[0]:rios[2], rios[1], rios[3]] = unrotated
    image[:, :, 3] = (image[:, :, 3] * blur_transparency).astype(np.uint8)
    return image


def alpha_composite_gpu(foreground, background):
    fg_gpu = cupy.array(foreground, dtype=cupy.float32)
    bg_gpu = cupy.array(background, dtype=cupy.float32)

    fg_rgb = fg_gpu[:, :, :3]
    fg_alpha = fg_gpu[:, :, 3] / 255
    bg_rgb = bg_gpu[:, :, :3]
    bg_alpha = bg_gpu[:, :, 3] / 255

    out_alpha = fg_alpha + bg_alpha * (1 - fg_alpha)
    out_alpha_mask = out_alpha > 0
    out_rgb = cupy.zeros_like(fg_rgb)

    for c in range(3):
        out_rgb[:, :, c] = (fg_rgb[:, :, c] * fg_alpha +
                            bg_rgb[:, :, c] * bg_alpha * (1 - fg_alpha))
        out_rgb[:, :, c] = cupy.where(out_alpha_mask,
                                      out_rgb[:, :, c] / out_alpha,
                                      out_rgb[:, :, c])

    out_image = cupy.dstack((out_rgb, out_alpha * 255)).astype(cupy.uint8)
    return out_image

def alpha_composite_cpu(foreground, background):
    fg = np.array(foreground, dtype=np.float32)
    bg = np.array(background, dtype=np.float32)

    print("bg max alpha:", np.max(background[:, :, 3]))
    print("fg max alpha:", np.max(foreground[:, :, 3]))

    fg_rgb = fg[:, :, :3]
    fg_alpha = fg[:, :, 3] / 255
    bg_rgb = bg[:, :, :3]
    bg_alpha = bg[:, :, 3] / 255

    out_alpha = fg_alpha + bg_alpha * (1 - fg_alpha)
    out_alpha_mask = out_alpha > 0
    out_rgb = np.zeros_like(fg_rgb)

    for c in range(3):
        out_rgb[:, :, c] = (
                fg_rgb[:, :, c] * fg_alpha +
                bg_rgb[:, :, c] * bg_alpha * (1 - fg_alpha)
        )
    epsilon = 1e-5
    out_rgb = np.divide(out_rgb, out_alpha[..., np.newaxis] + epsilon)

    out_image = np.dstack((out_rgb, np.ones_like(bg_alpha) * 255)).astype(np.uint8)
    return out_image


def generate_motion_blur_image(video_id, blur_strength, blur_transparency, frame_skip):
    frames_paths = sorted(sv.list_files_with_extensions(directory=get_foreground_temp_image_folder(video_id).__str__(),
                                                        extensions=["png"]))
    prev_frame = cv2.imread(frames_paths[0].__str__())
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2BGRA)
    _, prev_center_x, _, _, prev_center_y, _ = get_max_min_center_of_object(prev_frame)
    base_image = np.zeros_like(prev_frame)

    image_path = get_motion_blur_image(video_id, "motion_blur.png")
    cv2.imwrite(image_path, base_image)
    final_image = Image.open(image_path)
    background_array = get_background(video_id, len(frames_paths) - 1)
    background_array = cv2.cvtColor(background_array, cv2.COLOR_BGRA2RGBA)
    background = Image.fromarray(background_array)
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
    current_frame_path = frames_paths[current_frame - 1].__str__()
    to_be_blurred_frame = cv2.imread(current_frame_path, cv2.IMREAD_UNCHANGED)
    min_x, center_x, max_x, min_y, center_y, max_y = get_max_min_center_of_object(to_be_blurred_frame)
    while True:
        if current_frame >= len(frames_paths):
            break

        start = timer()
        print(current_frame)

        next_index = current_frame + 1 + frame_skip
        if next_index >= len(frames_paths):
            next_index = len(frames_paths) - 1
        next_frame_path = frames_paths[next_index].__str__()
        next_frame = cv2.imread(next_frame_path, cv2.IMREAD_UNCHANGED)
        next_min_x, next_center_x, next_max_x, next_min_y, next_center_y, next_max_y = get_max_min_center_of_object(
            next_frame)

        delta_x = center_x - next_center_x
        delta_y = center_y - next_center_y
        angle = np.degrees(np.arctan2(delta_x, delta_y)) + 90
        magnitude = np.sqrt(delta_x ** 2 + delta_y ** 2) / np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2) * 8
        magnitude = np.min([magnitude, 5])

        blurred_frame = create_blurred_frame_global_kernel(to_be_blurred_frame, magnitude, angle, min_x, min_y, max_x,
                                                           max_y)
        blurred_frame[:, :, 3] = (blurred_frame[:, :, 3] * blur_transparency).astype(np.uint8)

        blur_path = get_motion_blur_folder(video_id).joinpath(Path(os.path.basename(current_frame_path)).stem + ".png")
        cv2.imwrite(blur_path, blurred_frame)
        blur_image = Image.open(blur_path)
        final_image.alpha_composite(blur_image)
        end = timer()
        print("--- %s seconds ---" % (end - start))
        print("---------------------")

        current_frame += 1 + frame_skip
        min_x, center_x, max_x, min_y, center_y, max_y = next_min_x, next_center_x, next_max_x, next_min_y, next_center_y, next_max_y
        current_frame_path = next_frame_path
        to_be_blurred_frame = next_frame
    last_frame = Image.open(frames_paths[len(frames_paths) - 1].__str__())
    final_image.alpha_composite(last_frame)
    final_image.save(image_path.__str__())
    return image_path


def read_image(file_path):
    return cv2.imread(file_path, cv2.IMREAD_UNCHANGED)


def write_image(file_path, image):
    cv2.imwrite(file_path, image)


def read_images(file_paths):
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(read_image, file_paths))
    return images


def write_images(file_paths, images):
    with ThreadPoolExecutor() as executor:
        executor.map(write_image, file_paths, images)


def get_roi_gpu(image, stream):
    with stream:
        alpha_channel = image[..., 3]
        visible_pixels = cupy.argwhere(alpha_channel > 0)
        min_y, min_x = cupy.min(visible_pixels, axis=0).get()
        max_y, max_x = cupy.max(visible_pixels, axis=0).get()
    return [min_x, min_y, max_x, max_y]


def get_roi_cpu(image):
    alpha_channel = image[..., 3]
    visible_pixels = np.argwhere(alpha_channel > 0)
    min_y, min_x = np.min(visible_pixels, axis=0)
    max_y, max_x = np.max(visible_pixels, axis=0)
    return [min_x, min_y, max_x, max_y]


def get_center_gpu(roi, stream):
    with stream:
        center_x = ((roi[2] + roi[0]) / 2).item()
        center_y = ((roi[3] + roi[1]) / 2).item()
    return [center_x, center_y]

def get_center_cpu(roi):
    center_x = ((roi[2] + roi[0]) / 2)
    center_y = ((roi[3] + roi[1]) / 2)
    return np.array([center_x, center_y])


def get_delta_gpu(center, next_center, stream):
    with stream:
        delta_x = next_center[0] - center[0]
        delta_y = next_center[1] - center[1]
    return [delta_x, delta_y]

def get_delta_cpu(center, next_center):
    delta_x = next_center[0] - center[0]
    delta_y = next_center[1] - center[1]
    return [delta_x, delta_y]


def get_angle_gpu(delta, stream):
    with stream:
        angle = cupy.degrees(cupy.arctan2(delta[1], delta[0]))
    return cupy.round(angle, 1)

def average_out_angle_gpu(angles):
    angles = cupy.array(angles)
    avg_angles = cupy.convolve(angles, cupy.array([1/5,1/5,1/5,1/5,1/5], dtype=cupy.float32), mode='same')
    avg_angles = cupy.round(avg_angles, 1)
    return avg_angles


def get_angle_cpu(delta):
    return (np.degrees(np.arctan2(delta[0], delta[1])) + 360 + 180) % 360


def get_magnitude_gpu(delta, roi, stream):
    with stream:
        size = cupy.sqrt((roi[2] - roi[0]) ** 2 + (roi[3] - roi[1]) ** 2)
        distance = cupy.sqrt(delta[0] ** 2 + delta[1] ** 2)
        magnitude = float(distance / size) + 1
    return magnitude

def get_magnitude_cpu(delta, roi):
    size = np.sqrt((roi[2] - roi[0]) ** 2 + (roi[3] - roi[1]) ** 2)
    distance = np.sqrt(delta[0] ** 2 + delta[1] ** 2)
    return float(distance / size) + 1


def get_max_min_center_of_object_gpu(frame):
    alpha_channel = frame[..., 3]
    visible_pixels = cupy.argwhere(alpha_channel > 0)
    min_y, min_x = cupy.min(visible_pixels, axis=0).get()
    max_y, max_x = cupy.max(visible_pixels, axis=0).get()
    center_x = ((max_x + min_x) / 2).item()
    center_y = ((max_y + min_y) / 2).item()
    return int(min_x), int(center_x), int(max_x), int(min_y), int(center_y), int(max_y)


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
    kernel = kernel_list[movement - 1][direction - 1]
    return cv2.filter2D(frame, -1, kernel)


def create_multiple_instance_effect(video_id, output_path, instance_count, frame_skip, transparency_mode="uniform",
                                    transparency_strength=0.5, frame_offset=0):
    try:
        if frame_offset > 0:
            raise ValueError("Only negative offsets are allowed when starting from the last frame.")

        foreground_folder = get_foreground_temp_image_folder(video_id)
        background_folder = get_background_temp_image_folder(video_id)

        foreground_frames = sorted(Path(foreground_folder).glob("*.png"))
        background_frames = sorted(Path(background_folder).glob("*.png"))
        if not foreground_frames or not background_frames:
            raise ValueError("No frames found in the specified folders.")

        start_frame_index = len(foreground_frames) - 1 + frame_offset
        if start_frame_index < 0 or start_frame_index >= len(foreground_frames):
            raise ValueError("Frame offset is out of range.")

        background_frame = cv2.imread(str(background_frames[start_frame_index]), cv2.IMREAD_UNCHANGED)
        background_frame = cv2.cvtColor(background_frame, cv2.COLOR_BGR2BGRA)

        instance_count = min(instance_count - 1, (start_frame_index + 1) // frame_skip)
        if instance_count < 0:
            raise ValueError("Not enough frames available for the given instance count and frame skip.")

        alpha_values = []
        for i in range(instance_count):
            if transparency_mode == "uniform":
                alpha = int(255 * transparency_strength)
            elif transparency_mode == "gradient linear":
                alpha = max(1, min(255, int(255 * ((i + 1) / instance_count) * transparency_strength)))
            elif transparency_mode == "gradient quadratic":
                alpha = int(255 * ((i + 1) / instance_count) ** 0.5 * transparency_strength)
            alpha_values.append(alpha)

        # paralized loading of frames
        frame_indices = [-(instance_count * frame_skip) + (i * frame_skip) + start_frame_index for i in range(instance_count) if -(instance_count * frame_skip) + (i * frame_skip) + start_frame_index >= 0]
        frame_paths = list([foreground_frames[i] for i in frame_indices])
        foreground_images = read_images(frame_paths)

        for foreground_frame, alpha in zip(foreground_images, alpha_values):
            if foreground_frame is None:
                continue

            foreground_frame[:, :, 3] = (foreground_frame[:, :, 3] * (alpha / 255)).astype(np.uint8)
            mask = foreground_frame[:, :, 3] > 0
            foreground_alpha = foreground_frame[:, :, 3] / 255.0

            background_frame[:, :, :3][mask] = (
                foreground_frame[:, :, :3][mask] * foreground_alpha[mask, None] +
                background_frame[:, :, :3][mask] * (1 - foreground_alpha[mask, None])
            ).astype(np.uint8)
            background_frame[:, :, 3][mask] = np.maximum(background_frame[:, :, 3][mask], foreground_frame[:, :, 3][mask])

        last_foreground_frame = cv2.imread(str(foreground_frames[start_frame_index]), cv2.IMREAD_UNCHANGED)
        if last_foreground_frame is not None:
            mask = last_foreground_frame[:, :, 3] > 0
            background_frame[:, :, :3][mask] = last_foreground_frame[:, :, :3][mask]
            background_frame[:, :, 3][mask] = 255

        output_image = cv2.cvtColor(background_frame, cv2.COLOR_BGRA2RGBA)
        Image.fromarray(output_image).save(output_path, format="PNG")
        print(f"Multiple instance effect created with offset {frame_offset} and saved to {output_path}")

    except Exception as e:
        print(f"Error creating multiple instance effect: {e}")



def create_multiple_instance_effect_reversed(video_id, output_path, instance_count, frame_skip, transparency_mode="uniform",
                                    transparency_strength=0.5, frame_offset=0):
    try:
        foreground_folder = get_foreground_temp_image_folder(video_id)
        background_folder = get_background_temp_image_folder(video_id)

        background_frames = sorted(Path(background_folder).glob("*.png"))
        foreground_frames = sorted(Path(foreground_folder).glob("*.png"))
        max_offset = len(background_frames) - 1
        frame_offset = min(frame_offset, max_offset)

        background_frame = cv2.imread(str(background_frames[frame_offset]), cv2.IMREAD_UNCHANGED)
        background_frame = cv2.cvtColor(background_frame, cv2.COLOR_BGR2BGRA)

        foreground_frames = foreground_frames[frame_offset:]
        instance_count = min(instance_count - 1, len(foreground_frames) // frame_skip)
        if instance_count < 0:
            raise ValueError("Not enough frames available for the given instance count and frame skip.")

        alpha_values = []
        for i in range(instance_count, -1, -1):
            if i == 0:
                alpha = 255
            else:
                if transparency_mode == "uniform":
                    alpha = int(255 * transparency_strength)
                elif transparency_mode == "gradient linear":
                    alpha = max(1, min(255, int(255 * (1 - (i / instance_count)) * transparency_strength)))
                elif transparency_mode == "gradient quadratic":
                    offset = 0.7
                    progress = (i / instance_count) + offset
                    alpha = max(1, min(255, int(255 * (1 - (progress ** 0.5 - offset ** 0.5)) * transparency_strength)))
            alpha_values.append(alpha)

        frame_indices = [i * frame_skip for i in range(instance_count, -1, -1) if i * frame_skip < len(foreground_frames)]
        frame_paths = list([foreground_frames[i] for i in frame_indices])
        foreground_images = read_images(frame_paths)

        for foreground_frame, alpha in zip(foreground_images, alpha_values):
            if foreground_frame is None:
                continue

            foreground_frame[:, :, 3] = (foreground_frame[:, :, 3] * (alpha / 255)).astype(np.uint8)
            mask = foreground_frame[:, :, 3] > 0
            foreground_alpha = foreground_frame[:, :, 3] / 255.0

            background_frame[:, :, :3][mask] = (
                foreground_frame[:, :, :3][mask] * foreground_alpha[mask, None] +
                background_frame[:, :, :3][mask] * (1 - foreground_alpha[mask, None])
            ).astype(np.uint8)
            background_frame[:, :, 3][mask] = np.maximum(background_frame[:, :, 3][mask], foreground_frame[:, :, 3][mask])

        output_image = cv2.cvtColor(background_frame, cv2.COLOR_BGRA2RGBA)
        Image.fromarray(output_image).save(output_path, format="PNG")
        print(f"Multiple instance effect (reversed) with offset {frame_offset} created and saved to {output_path}")

    except Exception as e:
        print(f"Error creating reversed multiple instance effect: {e}")


def create_multiple_instance_effect_middle(video_id, output_path, instance_count, frame_skip, transparency_mode="uniform",
                                    transparency_strength=0.5, frame_offset=0):
    try:
        foreground_folder = get_foreground_temp_image_folder(video_id)
        background_folder = get_background_temp_image_folder(video_id)

        foreground_frames = sorted(Path(foreground_folder).glob("*.png"))
        background_frames = sorted(Path(background_folder).glob("*.png"))

        middle_index = len(foreground_frames) // 2
        reference_index = middle_index + frame_offset
        reference_index = max(0, min(reference_index, len(foreground_frames) - 1))

        background_frame_path = str(background_frames[reference_index])
        background_frame = cv2.imread(background_frame_path, cv2.IMREAD_UNCHANGED)
        background_frame = cv2.cvtColor(background_frame, cv2.COLOR_BGR2BGRA)

        left_indices = list(range(reference_index - frame_skip, -1, -frame_skip))
        right_indices = list(range(reference_index + frame_skip, len(foreground_frames), frame_skip))
        selected_indices = left_indices + right_indices
        selected_indices.sort(key=lambda x: abs(x - reference_index))
        selected_indices = selected_indices[:instance_count - 1]

        frame_paths = list([foreground_frames[i] for i in selected_indices])
        sorted_frames = read_images(frame_paths)

        # reverse insertion of frames
        for foreground_frame, frame_index in reversed(list(zip(sorted_frames, selected_indices))):
            if foreground_frame is None:
                continue

            distance_from_reference = abs(frame_index - reference_index)
            max_distance = max(reference_index, len(foreground_frames) - reference_index)

            if transparency_mode == "uniform":
                alpha = int(255 * transparency_strength)
            elif transparency_mode == "gradient linear":
                alpha = int(255 * (1 - (distance_from_reference / max_distance)) * transparency_strength)
                alpha = max(1, min(255, alpha))
            elif transparency_mode == "gradient quadratic":
                offset = 0.7
                progress = (distance_from_reference / max_distance) + offset
                alpha = int(255 * (1 - (progress ** 0.5 - offset ** 0.5)) * transparency_strength)
                alpha = max(1, min(255, alpha))

            foreground_frame[:, :, 3] = (foreground_frame[:, :, 3] * (alpha / 255)).astype(np.uint8)
            mask = foreground_frame[:, :, 3] > 0
            foreground_alpha = foreground_frame[:, :, 3]

            for c in range(3):
                background_frame[:, :, c] = np.where(
                    mask,
                    foreground_frame[:, :, c] * (foreground_alpha / 255) +
                    background_frame[:, :, c] * (1 - foreground_alpha / 255),
                    background_frame[:, :, c]
                ).astype(np.uint8)

            background_frame[:, :, 3] = np.where(
                mask,
                np.maximum(background_frame[:, :, 3], foreground_alpha),
                background_frame[:, :, 3]
            ).astype(np.uint8)

        reference_frame = cv2.imread(str(foreground_frames[reference_index]), cv2.IMREAD_UNCHANGED)
        mask = reference_frame[:, :, 3] > 0
        for c in range(3):
            background_frame[:, :, c] = np.where(
                mask,
                reference_frame[:, :, c] * (reference_frame[:, :, 3] / 255) +
                background_frame[:, :, c] * (1 - reference_frame[:, :, 3] / 255),
                background_frame[:, :, c]
            ).astype(np.uint8)
        background_frame[:, :, 3] = np.where(
            mask,
            reference_frame[:, :, 3],
            background_frame[:, :, 3]
        ).astype(np.uint8)

        output_image = cv2.cvtColor(background_frame, cv2.COLOR_BGRA2RGBA)
        Image.fromarray(output_image).save(output_path, format="PNG")
        print(f"Multiple instance effect with reference offset {frame_offset} created and saved to {output_path}")

    except Exception as e:
        print(f"Error creating multiple instance effect: {e}")


def create_action_line_effect(video_id, thickness, start_percentage, smoothing_factor=7, color="#FFFFFFFF"):
    # load foregrounds
    print(color)
    result_path = get_motion_blur_image(video_id, "action_line.png")
    frame_folder_path = get_foreground_temp_image_folder(video_id).__str__()
    frames_paths = sorted(sv.list_files_with_extensions(directory=frame_folder_path, extensions=["png"]))

    used_frame_paths = []
    last_frame_id = len(frames_paths) - 1
    for i in range(last_frame_id, -1, -1):
        used_frame_paths.insert(0, frames_paths[i].__str__())
    print(last_frame_id)
    used_frames = read_images(used_frame_paths)

    # calculate points
    with ThreadPoolExecutor() as executor:
        rois = list(executor.map(get_roi_cpu, used_frames))
        print("finished rois")
        centers = list(executor.map(get_center_cpu, rois))
        print("finished centers")
    print(0)
    # smooth points
    if smoothing_factor > 1:
        centers_smoothed = smooth_points_savgol(centers, window_length=smoothing_factor)
    else:
        centers_smoothed = centers
    smoothed_points = catmull_rom_spline(centers_smoothed, 20, 0.5)
    print(1)

    deltas = []
    # Calculate starting point
    for i in range(1, len(smoothed_points)):
        pt1 = smoothed_points[i - 1]
        pt2 = smoothed_points[i]
        deltas.append(get_delta_cpu(pt1, pt2))

    distances = []
    total_distance = 0
    for i in range(0, len(deltas) - 1):
        delta = deltas[i]
        distance = np.sqrt(delta[0] ** 2 + delta[1] ** 2)
        distances.append(distance)
        total_distance += distance

    start_point = float(total_distance / 100 * start_percentage)
    start_index = 0
    current_distance = 0.0
    for i in range(0, len(distances)-1):
        if start_point > current_distance:
            current_distance += distances[i]
        else:
            start_index = i
            break

    print("Total line distance: ", total_distance)
    print("Starting point: ", start_index, "/", len(smoothed_points))
    print("Wanted line length percentage: ", 1 - start_point / total_distance)
    print("Actual line length percentage: ", 1 - current_distance / total_distance)

    # create lines
    resulting_image = get_background(video_id, last_frame_id)

    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)
    a = min(float(int(color[7:9], 16)) / 255, 0.999)
    print(r, g, b, a, 1-a)

    overlay = resulting_image.copy()

    for i in range(start_index + 1, len(smoothed_points)):
        pt1 = smoothed_points[i - 1]
        pt2 = smoothed_points[i]
        cv2.line(overlay, pt1, pt2, (b, g, r, a), thickness)

    cv2.addWeighted(overlay, a, resulting_image, 1-a, 0, resulting_image)

    # Sicherstellen, dass background und foreground 4 Kanäle haben
    if resulting_image.shape[2] == 3:
        print("AAAAAAAAAAAAAAAAAAAA")
        h, w = resulting_image.shape[:2]
        alpha = np.full((h, w, 1), 255, dtype=np.uint8)
        resulting_image = np.concatenate((resulting_image, alpha), axis=2)

    resulting_image = alpha_composite_cpu(cv2.cvtColor(used_frames[last_frame_id], cv2.COLOR_BGR2BGRA), resulting_image)

    cv2.imwrite(result_path.__str__(), resulting_image)

def draw_vectors(x_size, y_size, vectors):
    image = np.zeros((x_size, y_size, 4), dtype=np.uint8)
    for start, end in vectors:
        cv2.line(image, start, end, (0, 0, 0, 255), 1)
    return image

def catmull_rom_spline(points, count=20, alpha=0.5):
    def tj(ti, pi, pj):
        dist = np.linalg.norm(pj - pi)
        return ti + (dist ** alpha if dist > 1e-5 else 1e-5)  # Vermeide Null-Division

    result = []
    for i in range(1, len(points) - 2):
        p0, p1, p2, p3 = points[i-1], points[i], points[i+1], points[i+2]
        t0 = 0
        t1 = tj(t0, p0, p1)
        t2 = tj(t1, p1, p2)
        t3 = tj(t2, p2, p3)

        for t in np.linspace(t1, t2, count):
            try:
                A1 = (t1 - t)/(t1 - t0) * p0 + (t - t0)/(t1 - t0) * p1
                A2 = (t2 - t)/(t2 - t1) * p1 + (t - t1)/(t2 - t1) * p2
                A3 = (t3 - t)/(t3 - t2) * p2 + (t - t2)/(t3 - t2) * p3

                B1 = (t2 - t)/(t2 - t0) * A1 + (t - t0)/(t2 - t0) * A2
                B2 = (t3 - t)/(t3 - t1) * A2 + (t - t1)/(t3 - t1) * A3

                C = (t2 - t)/(t2 - t1) * B1 + (t - t1)/(t2 - t1) * B2
                if np.all(np.isfinite(C)):
                    result.append(C.astype(int))
            except ZeroDivisionError:
                continue  # Ignoriere Problem-Segmente

    return np.array(result)

def smooth_points_savgol(points, window_length=7, polyorder=2):
    points = np.array(points, dtype=np.float32)
    x = savgol_filter(points[:, 0], window_length, polyorder)
    y = savgol_filter(points[:, 1], window_length, polyorder)
    return np.stack((x, y), axis=-1)
