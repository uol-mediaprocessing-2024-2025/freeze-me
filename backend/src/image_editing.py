import functools
import io
import os
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from timeit import default_timer as timer

import cupy
import cupyx
import cv2
import numpy as np
import psutil
import scipy
import supervision as sv
import torch
from PIL import Image
from fastapi import UploadFile
from scipy.signal import savgol_filter

from path_manager import get_background_image, get_background_temp_image_path, get_foreground_parent_folder, \
    get_frame_path
from path_manager import get_foreground_temp_image_folder
from path_manager import get_motion_blur_image
from project_data import set_motion_blur_metadata, get_motion_blur_data, get_background_type, BackgroundType

max_kernel_size = 60
min_movement = 1
max_movement = 5
angle_range = 10

kernel_list = np.zeros((max_movement, np.floor(180 / angle_range).astype(int)), dtype='object')
kernel_list[:, :] = None


def get_device():
    return "cpu"


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
    background_path = get_frame_path(video_id, frame_id)
    background = cv2.imread(background_path, cv2.IMREAD_UNCHANGED)
    return cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)


def get_transparent_background(video_id, frame_id):
    frame_path = get_background_temp_image_path(video_id, frame_id)
    frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
    return np.zeros_like(frame)


async def create_motion_blur_image(video_id, blur_strength, blur_transparency, frame_skip):
    # device = get_device()
    device = "cpu"
    motion_blur_data = get_motion_blur_data(video_id)
    if not motion_blur_data:
        # If data doesn't exist, this is the first time generating motion blur image
        generate_blur = True
        motion_blur_data = [0, 0, 0]
    else:
        generate_blur = motion_blur_data[0] != blur_strength or motion_blur_data[1] != blur_transparency or \
                        motion_blur_data[2] != frame_skip
    if device == "cuda":
        path = gpu_motion_blur(video_id, blur_strength, blur_transparency, frame_skip, True)
    else:
        path = cpu_motion_blur(video_id, blur_strength, blur_transparency, frame_skip, True)

    # update motion blur data
    motion_blur_data[0] = blur_strength
    motion_blur_data[1] = blur_transparency
    motion_blur_data[2] = frame_skip
    set_motion_blur_metadata(video_id, motion_blur_data)

    return path


def gpu_motion_blur(video_id, blur_strength, blur_transparency, frame_skip, generate_blur):
    # Get needed frames: regular if new blur is generated and already blurred frames if blur isn't necessary
    result_path = get_motion_blur_image(video_id, "motion_blur.png")
    object_count = len(os.listdir(get_foreground_parent_folder(video_id)))
    resulting_image_gpu = None

    warmup_cupy_kernels()
    while object_count > 0:
        object_count -= 1
        if not generate_blur:
            return result_path

        frame_path = get_foreground_temp_image_folder(video_id, object_count).__str__()
        frames_paths = sorted(sv.list_files_with_extensions(directory=frame_path, extensions=["png"]))
        used_frame_paths = []

        load_start = timer()
        last_frame_id = len(frames_paths) - 1
        for i in range(last_frame_id, -1, -(1 + frame_skip)):
            used_frame_paths.insert(0, frames_paths[i].__str__())
        print(f"Last frame id: {last_frame_id}")
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

        if resulting_image_gpu is None:
            resulting_image = get_background(video_id, last_frame_id)
            resulting_image_gpu = cupy.array(resulting_image, dtype=cupy.uint8)

        # generate image batch for batch
        for i in range(0, len(used_frames), batch_size):
            # load frames into gpu
            batch = used_frames[i:i + batch_size]

            object_gpu_batch = generate_blur_for_object_gpu(batch, blur_strength, blur_transparency, frame_skip,
                                                            generate_blur)

            comp_start = timer()
            for j in range(0, len(object_gpu_batch)):
                resulting_image_gpu = alpha_composite_gpu(object_gpu_batch[j], resulting_image_gpu)

            comp_end = timer()
            print("--- Composite: %s seconds ---" % (comp_end - comp_start))
            print("---------------------")
        del object_gpu_batch

    # save and return
    resulting_image = cupy.asnumpy(resulting_image_gpu)
    cv2.imwrite(result_path.__str__(), resulting_image)
    cupy.get_default_memory_pool().free_all_blocks()
    return result_path


def generate_blur_for_object_gpu(batch, blur_strength, blur_transparency, frame_skip, generate_blur):
    gpu_start = timer()
    batch_gpu = [cupy.array(image, dtype=cupy.uint8) for image in batch]
    gpu_end = timer()
    print("--- Transfer to GPU: %s seconds ---" % (gpu_end - gpu_start))
    print("---------------------")
    if not generate_blur:
        return batch_gpu

    start = timer()
    with ThreadPoolExecutor(max_workers=min(8, len(batch_gpu))) as executor:
        rois = list(executor.map(get_roi_gpu, batch_gpu))
        valid_rois = []
        valid_batch_gpu = []
        for i in range(0, len(batch_gpu) - 1):
            roi = rois[i]
            if roi is not None:
                valid_rois.append(roi)
                valid_batch_gpu.append(batch_gpu[i])
        batch_gpu = valid_batch_gpu
        rois = valid_rois
        del valid_rois

        print("finished rois")
        centers = list(executor.map(get_center_gpu, rois))
        print("finished centers")

        final_centers = [centers[len(centers) - 1]]
        valid_batch_gpu = [batch_gpu[len(centers) - 1]]
        valid_rois = [rois[len(centers) - 1]]
        last_center_index = len(centers) - 1
        print(f"batch_gpu_length: {len(batch_gpu)}")
        for j in range(len(centers) - 2, -1, -1):
            last_center = final_centers[len(final_centers) - 1]
            potential_center = centers[j]
            magnitude = get_magnitude_gpu(get_delta_gpu(last_center, potential_center), rois[last_center_index])
            print(f"magnitude: {magnitude}, last_center_index: {last_center_index}, j: {j}")
            if magnitude > 0.3:
                final_centers.append(centers[j])
                last_center_index = j
                valid_batch_gpu.append(batch_gpu[j])
                valid_rois.append(rois[j])

        final_centers.reverse()
        valid_batch_gpu.reverse()
        valid_rois.reverse()
        centers, batch_gpu, rois = final_centers, valid_batch_gpu, valid_rois
        del valid_batch_gpu, final_centers, valid_rois

        deltas = list(executor.map(get_delta_gpu, centers[:-1], centers[1:]))
        print("finished deltas")
        angles = list(executor.map(get_angle_gpu, deltas))
        print("finished angles")
        magnitudes = list(executor.map(get_magnitude_gpu, deltas, rois[:-1]))
        print("finished magnitudes")
        cupy.get_default_memory_pool().free_all_blocks()

    prep_time = timer()
    print("--- Preparing: %s seconds ---" % (prep_time - start))
    print("---------------------")
    print(f"batch_gpu_length: {len(batch_gpu)}")

    with ThreadPoolExecutor() as executor:
        streams = [cupy.cuda.Stream() for _ in batch_gpu[:-1]]
        gpu_blur_image_with_defaults = functools.partial(
            gpu_blur_image,
            blur_strength=blur_strength,
            blur_transparency=blur_transparency,
            frame_skip=frame_skip,
            show=False
        )
        blurred_batch_gpu = list(
            executor.map(gpu_blur_image_with_defaults, batch_gpu[:-1], angles, magnitudes, rois[:-1], centers[:-1],
                         streams))
        [stream.synchronize() for stream in streams]
        del streams
        cupy.get_default_memory_pool().free_all_blocks()

    blurred_batch_gpu.append(batch_gpu[-1])
    print(f"blurred_batch_gpu_length: {len(blurred_batch_gpu)}")
    blur_time = timer()
    print("--- Blurring: %s seconds ---" % (blur_time - prep_time))
    print("---------------------")
    return blurred_batch_gpu


def generate_blur_for_object_cpu(batch, blur_strength, blur_transparency, frame_skip, generate_blur):
    if not generate_blur:
        return batch

    start = timer()
    with ThreadPoolExecutor(max_workers=min(8, len(batch))) as executor:
        rois = list(executor.map(get_roi_cpu, batch))
        valid_rois = []
        valid_batch_gpu = []
        for i in range(0, len(batch) - 1):
            roi = rois[i]
            if roi is not None:
                valid_rois.append(roi)
                valid_batch_gpu.append(batch[i])
        batch = valid_batch_gpu
        rois = valid_rois
        del valid_rois

        print("finished rois")
        centers = list(executor.map(get_center_cpu, rois))
        print("finished centers")

        final_centers = [centers[len(centers) - 1]]
        valid_batch_gpu = [batch[len(centers) - 1]]
        valid_rois = [rois[len(centers) - 1]]
        last_center_index = len(centers) - 1
        print(f"batch_gpu_length: {len(batch)}")
        for j in range(len(centers) - 2, -1, -1):
            last_center = final_centers[len(final_centers) - 1]
            potential_center = centers[j]
            magnitude = get_magnitude_cpu(get_delta_cpu(last_center, potential_center), rois[last_center_index])
            print(f"magnitude: {magnitude}, last_center_index: {last_center_index}, j: {j}")
            if magnitude > 0.3:
                final_centers.append(centers[j])
                last_center_index = j
                valid_batch_gpu.append(batch[j])
                valid_rois.append(rois[j])

        final_centers.reverse()
        valid_batch_gpu.reverse()
        valid_rois.reverse()
        centers, batch, rois = final_centers, valid_batch_gpu, valid_rois
        del valid_batch_gpu, final_centers, valid_rois

        deltas = list(executor.map(get_delta_cpu, centers[:-1], centers[1:]))
        print("finished deltas")
        angles = list(executor.map(get_angle_cpu, deltas))
        print("finished angles")
        magnitudes = list(executor.map(get_magnitude_cpu, deltas, rois[:-1]))
        print("finished magnitudes")
    prep_time = timer()
    print("--- Preparing: %s seconds ---" % (prep_time - start))
    print("---------------------")
    print(f"batch_gpu_length: {len(batch)}")

    with ThreadPoolExecutor() as executor:
        cpu_blur_image_with_defaults = functools.partial(
            cpu_blur_image,
            blur_strength=blur_strength,
            blur_transparency=blur_transparency,
            frame_skip=frame_skip,
        )
        blurred_batch_gpu = list(
            executor.map(cpu_blur_image_with_defaults, batch[:-1], angles, magnitudes, rois[:-1], centers[:-1]))
    blurred_batch_gpu.append(batch[-1])
    print(f"blurred_batch_gpu_length: {len(blurred_batch_gpu)}")
    blur_time = timer()
    print("--- Blurring: %s seconds ---" % (blur_time - prep_time))
    print("---------------------")
    return blurred_batch_gpu


def cpu_motion_blur(video_id, blur_strength, blur_transparency, frame_skip, generate_blur):
    # Get needed frames: regular if new blur is generated and already blurred frames if blur isn't necessary
    result_path = get_motion_blur_image(video_id, "motion_blur.png")
    object_count = len(os.listdir(get_foreground_parent_folder(video_id)))
    resulting_image_gpu = None

    while object_count > 0:
        object_count -= 1
        if not generate_blur:
            return result_path

        frame_path = get_foreground_temp_image_folder(video_id, object_count).__str__()
        frames_paths = sorted(sv.list_files_with_extensions(directory=frame_path, extensions=["png"]))
        used_frame_paths = []

        load_start = timer()
        last_frame_id = len(frames_paths) - 1
        for i in range(last_frame_id, -1, -(1 + frame_skip)):
            used_frame_paths.insert(0, frames_paths[i].__str__())
        print(f"Last frame id: {last_frame_id}")
        used_frames = read_images(used_frame_paths)
        load_end = timer()
        print("--- Loading: %s seconds ---" % (load_end - load_start))
        print("---------------------")
        print("Loaded: " + str(len(used_frames)) + " frames.")

        # Calculate how many images can be in a batch for the gpu
        ram = psutil.virtual_memory()
        free_mem = ram.free / 1024 ** 2
        byte_size = used_frames[0].size
        mb_size = byte_size / 1024 ** 2
        max_images = round(free_mem / mb_size)
        print(f"Free RAM: {free_mem:.2f} MB")
        print(f"Maximum amount of MB per image: {mb_size}")
        print(f"Maximum amount of images in RAM: {max_images}")

        # Set batch size lower than maximum possible just in case
        batch_size = max_images / 2
        batch_size = round(batch_size * 0.75)
        batch_size = len(used_frames) if batch_size > len(used_frames) else batch_size

        if resulting_image_gpu is None:
            resulting_image = get_background(video_id, last_frame_id)
            resulting_image_gpu = np.array(resulting_image, dtype=np.uint8)

        # generate image batch for batch
        for i in range(0, len(used_frames), batch_size):
            # load frames into gpu
            batch = used_frames[i:i + batch_size]

            object_gpu_batch = generate_blur_for_object_cpu(batch, blur_strength, blur_transparency, frame_skip,
                                                            generate_blur)

            comp_start = timer()
            for j in range(0, len(object_gpu_batch)):
                resulting_image_gpu = alpha_composite_cpu(object_gpu_batch[j], resulting_image_gpu)

            comp_end = timer()
            print("--- Composite: %s seconds ---" % (comp_end - comp_start))
            print("---------------------")
        del object_gpu_batch

    # save and return
    cv2.imwrite(result_path.__str__(), resulting_image_gpu)
    return result_path


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
        padded = cupy.pad(stretched, pad_width=((0, 0), (kernel_size, kernel_size), (0, 0)), mode='constant',
                          constant_values=0)

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
        image[rois[1]:rois[3], rois[0]:rois[2]] = old_crop_mask  # deletes unblurred input
        image[new_y1:new_y2, new_x1:new_x2] = derotated_cropped  # adds blurred input

        # Add transparency
        image[:, :, 3] = (image[:, :, 3] * blur_transparency).astype(np.uint8)

    return image


def cpu_blur_image(image, angle, magnitude, rois, center, blur_strength, blur_transparency, frame_skip):
    # create kernel based on blur_strength, magnitude and frame_skip
    kernel_size = int(magnitude * 15 + blur_strength * 15 + frame_skip * 5)
    kernel = np.zeros((1, kernel_size), dtype=np.float32)
    kernel[0] = 1.0 / kernel_size

    # Crop ROI
    cropped = image[rois[1]:rois[3], rois[0]:rois[2]]
    center_y = center[1]
    center_x = center[0]

    # Rotate ROI
    rotated = scipy.ndimage.rotate(cropped, angle, reshape=True, mode='constant', cval=0)

    # Stretch and blur
    stretched = scipy.ndimage.zoom(rotated, (0.9, magnitude, 1), order=1)
    padded = np.pad(stretched, pad_width=((0, 0), (kernel_size, kernel_size), (0, 0)), mode='constant',
                    constant_values=0)

    result = np.zeros_like(padded)
    for channel in range(4):  # Loop over RGBA channels
        result[:, :, channel] = scipy.ndimage.convolve(padded[:, :, channel], kernel, mode='constant', cval=0)

    # Rotate back
    derotated = scipy.ndimage.rotate(result, -angle, reshape=True, mode='constant', cval=0)

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
    image[rois[1]:rois[3], rois[0]:rois[2]] = old_crop_mask  # deletes unblurred input
    image[new_y1:new_y2, new_x1:new_x2] = derotated_cropped  # adds blurred input

    # Add transparency
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

    fg_rgb = fg[:, :, :3]
    fg_alpha = fg[:, :, 3] / 255
    bg_rgb = bg[:, :, :3]
    bg_alpha = bg[:, :, 3] / 255

    out_alpha = fg_alpha + bg_alpha * (1 - fg_alpha)
    out_alpha_mask = out_alpha > 0
    out_rgb = np.zeros_like(fg_rgb)

    for c in range(3):
        out_rgb[:, :, c] = (fg_rgb[:, :, c] * fg_alpha +
                            bg_rgb[:, :, c] * bg_alpha * (1 - fg_alpha))
        out_rgb[:, :, c] = np.where(out_alpha_mask,
                                    out_rgb[:, :, c] / out_alpha,
                                    out_rgb[:, :, c])

    out_image = np.dstack((out_rgb, out_alpha * 255)).astype(np.uint8)
    return out_image


def read_image(file_path):
    return cv2.imread(file_path, cv2.IMREAD_UNCHANGED)


def write_image(file_path, image):
    cv2.imwrite(file_path, image)


def read_images(file_paths):
    with ThreadPoolExecutor(max_workers=3) as executor:
        images = list(executor.map(read_image, file_paths))
    return images


def write_images(file_paths, images):
    with ThreadPoolExecutor() as executor:
        executor.map(write_image, file_paths, images)


def warmup_cupy_kernels():
    dummy = cupy.zeros((10, 10), dtype=cupy.uint8)
    dummy[0][0] = 2
    dummy[0][3] = 5
    mask = cupy.argwhere(dummy > 0)
    cupy.min(cupy.argwhere(mask), axis=0)
    cupy.max(cupy.argwhere(mask), axis=0)
    cupy.any(mask)
    del mask, dummy


def get_roi_gpu(image):
    alpha_channel = image[..., 3]
    visible_pixels = cupy.argwhere(alpha_channel > 0)
    if not cupy.any(visible_pixels):
        return None
    min_val = cupy.min(visible_pixels, axis=0)
    max_val = cupy.max(visible_pixels, axis=0)
    return [min_val[1], min_val[0], max_val[1], max_val[0]]


def get_roi_cpu(image):
    alpha_channel = image[..., 3]
    visible_pixels = np.argwhere(alpha_channel > 0)
    if not np.any(visible_pixels):
        return None
    min_y, min_x = np.min(visible_pixels, axis=0)
    max_y, max_x = np.max(visible_pixels, axis=0)
    return [min_x, min_y, max_x, max_y]


def get_center_gpu(roi):
    center_x = ((roi[2] + roi[0]) / 2).item()
    center_y = ((roi[3] + roi[1]) / 2).item()
    return [center_x, center_y]


def get_center_cpu(roi):
    center_x = ((roi[2] + roi[0]) / 2)
    center_y = ((roi[3] + roi[1]) / 2)
    return np.array([center_x, center_y])


def get_delta_gpu(center, next_center):
    delta_x = next_center[0] - center[0]
    delta_y = next_center[1] - center[1]
    return [delta_x, delta_y]


def get_delta_cpu(center, next_center):
    delta_x = next_center[0] - center[0]
    delta_y = next_center[1] - center[1]
    return [delta_x, delta_y]


def get_angle_gpu(delta):
    angle = cupy.degrees(cupy.arctan2(delta[1], delta[0]))
    return cupy.round(angle, 1)


def average_out_angle_gpu(angles):
    angles = cupy.array(angles)
    avg_angles = cupy.convolve(angles, cupy.array([1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5], dtype=cupy.float32), mode='same')
    avg_angles = cupy.round(avg_angles, 1)
    return avg_angles


def get_angle_cpu(delta):
    angle = np.degrees(np.arctan2(delta[1], delta[0]))
    return np.round(angle, 1)


def get_magnitude_gpu(delta, roi):
    size = cupy.sqrt((roi[2] - roi[0]) ** 2 + (roi[3] - roi[1]) ** 2)
    distance = cupy.sqrt(delta[0] ** 2 + delta[1] ** 2)
    magnitude = float(distance / size)
    return magnitude


def get_magnitude_cpu(delta, roi):
    size = np.sqrt((roi[2] - roi[0]) ** 2 + (roi[3] - roi[1]) ** 2)
    distance = np.sqrt(delta[0] ** 2 + delta[1] ** 2)
    return float(distance / size) + 1


def create_multiple_instance_effect_unified(
        video_id,
        output_path,
        instance_count,
        frame_skip,
        direction="last",  # "first", "middle"
        transparency_mode="uniform",
        transparency_strength=0.5,
        frame_offset=0
):
    # Hintergrundbild vorbereiten
    bg_img = None
    object_count = len(os.listdir(get_foreground_parent_folder(video_id)))

    while object_count > 0:
        object_count -= 1
        try:
            # Setup
            fg_folder = get_foreground_temp_image_folder(video_id, object_count)

            fg_frames = sorted(Path(fg_folder).glob("*.png"))
            if not fg_frames:
                raise ValueError("No frames found in foreground/background folders.")

            total_frames = len(fg_frames)

            # Bestimme Referenzindex für Hintergrundbild
            if direction == "last":
                ref_idx = total_frames - 1 + frame_offset
                ref_idx = max(0, min(ref_idx, total_frames - 1))
            elif direction == "middle":
                ref_idx = max(0, min((total_frames // 2) + frame_offset, total_frames - 1))
            else:
                ref_idx = min(frame_offset, total_frames - 1)

            if bg_img is None:
                bg_img = get_background(video_id, ref_idx)

            # Bestimme Frame-Indices je nach Richtung
            if direction == "last":
                available = (ref_idx + 1) // frame_skip
                count = min(instance_count - 1, available)
                indices = [ref_idx - i * frame_skip for i in range(count)][::-1]
                indices.append(ref_idx)
            elif direction == "middle":
                left = list(range(ref_idx - frame_skip, -1, -frame_skip))
                right = list(range(ref_idx + frame_skip, total_frames, frame_skip))
                mixed = sorted(left + right, key=lambda x: abs(x - ref_idx))[:instance_count - 1]
                indices = mixed[::-1]
                indices.append(ref_idx)
            else:
                available = (total_frames - ref_idx) // frame_skip
                count = min(instance_count - 1, available)
                indices = [ref_idx + i * frame_skip for i in range(count)]
                indices = indices[::-1]
                indices.append(ref_idx)

            # Lade Bilder
            fg_images = read_images([fg_frames[i] for i in indices])

            max_distance = max([abs(i - ref_idx) for i in indices]) or 1

            # Effekt anwenden
            for img, idx in zip(fg_images, indices):
                if img is None:
                    continue

                dist = abs(idx - ref_idx)
                if transparency_mode == "uniform":
                    alpha = int(255 * transparency_strength)
                elif transparency_mode == "gradient linear":
                    alpha = int(255 * (1 - dist / max_distance) * transparency_strength)
                elif transparency_mode == "gradient quadratic":
                    offset = 0.7
                    progress = (dist / max_distance) + offset
                    alpha = int(255 * (1 - (progress ** 0.5 - offset ** 0.5)) * transparency_strength)
                else:
                    raise ValueError("Invalid transparency mode.")

                alpha = max(1, min(255, alpha))
                img[:, :, 3] = (img[:, :, 3] * (alpha / 255)).astype(np.uint8)

                # Blending
                mask = img[:, :, 3] > 0
                fg_alpha = img[:, :, 3] / 255.0

                bg_img[:, :, :3][mask] = (
                        img[:, :, :3][mask] * fg_alpha[mask, None] +
                        bg_img[:, :, :3][mask] * (1 - fg_alpha[mask, None])
                ).astype(np.uint8)

                bg_img[:, :, 3][mask] = np.maximum(bg_img[:, :, 3][mask], img[:, :, 3][mask])

        except Exception as e:
            print(f"Error: {e}")

    # Ausgabe speichern
    out = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2RGBA)
    Image.fromarray(out).save(output_path, format="PNG")
    print(f"Effect ({direction}) created at {output_path}")


def create_action_line_effect(video_id, thickness, start_percentage, smoothing_factor=7, color="#FFFFFFFF"):
    # load foregrounds
    result_path = get_motion_blur_image(video_id, "action_line.png")
    object_count = len(os.listdir(get_foreground_parent_folder(video_id)))
    frame_folder_paths = []
    frames_paths = []
    last_frame_id = 0
    for i in range(0, object_count):
        frame_folder_path = get_foreground_temp_image_folder(video_id, i).__str__()
        frame_folder_paths.append(frame_folder_path)
        frames_paths = sorted(sv.list_files_with_extensions(directory=frame_folder_path, extensions=["png"]))
        frame_count = len(frames_paths) - 1
        if frame_count > last_frame_id:
            last_frame_id = frame_count

    print(f"last_frame_id: {last_frame_id}")
    resulting_image = get_background(video_id, last_frame_id)
    while object_count > 0:
        used_frame_paths = []
        for i in range(len(frames_paths) - 1, -1, -1):
            used_frame_paths.insert(0, frames_paths[i].__str__())
        print(f"Generating effect for object {object_count - 1}")
        used_frames = read_images(used_frame_paths)

        # calculate points
        with ThreadPoolExecutor() as executor:
            rois = list(executor.map(get_roi_cpu, used_frames))
            valid_rois = []
            for i in range(0, len(used_frames) - 1):
                roi = rois[i]
                if roi is not None:
                    valid_rois.append(roi)

            print(f"removed {len(rois) - len(valid_rois)} images from batch where no segmentation was found")
            print("finished rois")
            centers = list(executor.map(get_center_cpu, valid_rois))
            print("finished centers")
        # smooth points

        final_centers = [centers[len(centers) - 1]]
        last_center_index = len(centers) - 1
        for j in range(len(centers) - 2, -1, -1):
            last_center = final_centers[len(final_centers) - 1]
            potential_center = centers[j]
            magnitude = get_magnitude_cpu(get_delta_cpu(last_center, potential_center), rois[last_center_index])
            if magnitude > 0.25:
                final_centers.append(centers[j])

        final_centers.reverse()

        if smoothing_factor > 1:
            centers_smoothed = smooth_points_savgol(final_centers, window_length=smoothing_factor)
        else:
            centers_smoothed = final_centers
        smoothed_points = catmull_rom_spline(centers_smoothed, 20, 0.5)

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
        for i in range(0, len(distances) - 1):
            if start_point > current_distance:
                current_distance += distances[i]
            else:
                start_index = i
                break

        print("Total line distance: ", total_distance)
        print("Starting point: ", start_index, "/", len(smoothed_points))
        print("Wanted line length percentage: ", 1 - start_point / total_distance)
        print("Actual line length percentage: ", 1 - current_distance / total_distance)

        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        a = min(float(int(color[7:9], 16)) / 255, 0.999)

        overlay = resulting_image.copy()

        for i in range(start_index + 1, len(smoothed_points)):
            pt1 = smoothed_points[i - 1]
            pt2 = smoothed_points[i]
            cv2.line(overlay, pt1, pt2, (b, g, r, a), thickness)

        cv2.addWeighted(overlay, a, resulting_image, 1 - a, 0, resulting_image)

        # Sicherstellen, dass background und foreground 4 Kanäle haben
        if resulting_image.shape[2] == 3:
            h, w = resulting_image.shape[:2]
            alpha = np.full((h, w, 1), 255, dtype=np.uint8)
            resulting_image = np.concatenate((resulting_image, alpha), axis=2)

        resulting_image = alpha_composite_cpu(cv2.cvtColor(used_frames[len(used_frames) - 1], cv2.COLOR_BGR2BGRA),
                                              resulting_image)

        object_count -= 1
        if object_count > 0:
            frame_folder_path = get_foreground_temp_image_folder(video_id, object_count - 1).__str__()
            frames_paths = sorted(sv.list_files_with_extensions(directory=frame_folder_path, extensions=["png"]))

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
        p0, p1, p2, p3 = points[i - 1], points[i], points[i + 1], points[i + 2]
        t0 = 0
        t1 = tj(t0, p0, p1)
        t2 = tj(t1, p1, p2)
        t3 = tj(t2, p2, p3)

        for t in np.linspace(t1, t2, count):
            try:
                A1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
                A2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
                A3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3

                B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
                B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3

                C = (t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2
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
