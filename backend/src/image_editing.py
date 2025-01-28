import io
import os
from pathlib import Path
from timeit import default_timer as timer

import cupy
import cupyx
from cupyx.scipy import ndimage
import cv2
import numpy as np
import scipy
import supervision as sv
import torch
from PIL import Image
from fastapi import UploadFile
import matplotlib.pyplot as plt

from path_manager import get_background_image
from path_manager import get_background_temp_image_folder
from path_manager import get_foreground_temp_image_folder
from path_manager import get_motion_blur_folder
from path_manager import get_motion_blur_image
from project_data import set_motion_blur_metadata, get_motion_blur_data

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
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if len(image.shape)==3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    cv2.imwrite(final_image_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    path.unlink()
    return final_image_path


async def create_motion_blur_image(video_id, blur_strength, blur_transparency, frame_skip):
    device = torch.device("cuda" if torch.cuda.is_available() and cupy.cuda.is_available() else "cpu")
    motion_blur_data = get_motion_blur_data(video_id)
    if not motion_blur_data:
        # If data doesn't exist, this is the first time generating motion blur image
        generate_blur = True
        motion_blur_data = [0, 0, 0]
    else:
        generate_blur = motion_blur_data[0] != blur_strength or motion_blur_data[1] != blur_transparency or motion_blur_data[2] != frame_skip
    device = "cpu"
    if device.__eq__("cuda"):
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
    frames_paths = sorted(sv.list_files_with_extensions(directory=frame_path,extensions=["png"]))
    used_frames = []
    load_start = timer()
    for i in range(0, len(frames_paths), 1 + frame_skip):
        used_frames.append(cv2.imread(frames_paths[i].__str__(), cv2.IMREAD_UNCHANGED))

    load_end = timer()
    print("--- Loading: %s seconds ---" % (load_end - load_start))
    print("---------------------")

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

    # Copy background to result
    background_path = get_motion_blur_image(video_id, "background.png")
    background = cv2.imread(background_path, cv2.IMREAD_UNCHANGED)
    cv2.imwrite(result_path.__str__(), background)

    resulting_image = cv2.imread(result_path.__str__(), cv2.IMREAD_UNCHANGED)
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
            blurred_batch_gpu = []
            centers = []
            rios = []
            # calculate necessary data
            for frame in batch_gpu:
                min_x, center_x, max_x, min_y, center_y, max_y = get_max_min_center_of_object_gpu(frame)
                centers.append([center_x, center_y])
                rios.append([min_x, min_y, max_x, max_y])
            prep_time = timer()
            print("--- Preparing: %s seconds ---" % (prep_time - start))
            print("---------------------")
            for b in range(0, len(batch_gpu)):
                # calculate angle and magnitude for all but the last frame
                if b < len(batch_gpu) - 1:
                    delta_x = centers[b + 1][0] - centers[b][0]
                    delta_y = centers[b + 1][1] - centers[b][1]
                    angle = (cupy.degrees(cupy.arctan2(delta_x, delta_y)) + 360 + 180) % 360
                    size = cupy.sqrt((rios[b][2] - rios[b][0]) ** 2 + (rios[b][3] - rios[b][1]) ** 2)
                    distance = cupy.sqrt(delta_x ** 2 + delta_y ** 2)
                    magnitude = float(distance / size) + 1
                    blurred_batch_gpu.append(gpu_blur_image(batch_gpu[b], blur_strength, blur_transparency, frame_skip,
                                                          angle, magnitude, rios[b], centers[b], False))
                else:
                    blurred_batch_gpu.append(batch_gpu[b])
            blur_time = timer()
            print("--- Blurring: %s seconds ---" % (blur_time - prep_time))
            print("---------------------")
            # save blurred frames
            image_num = 0
            for b in range(i, batch_size, 1 + frame_skip):
                frame_name = str(b).zfill(5) + ".png"
                blur_path = get_motion_blur_folder(video_id).joinpath(frame_name).__str__()
                cv2.imwrite(blur_path, blurred_batch_gpu[image_num].get())
                image_num += 1
            save_time = timer()
            print("--- Saving: %s seconds ---" % (save_time - blur_time))
            print("---------------------")
        else:
            blurred_batch_gpu = batch_gpu
        # composite new image
        comp_start = timer()
        for j in range(0, len(batch_gpu)):
            resulting_image_gpu = alpha_composite_gpu(blurred_batch_gpu[j], resulting_image_gpu)

        comp_end = timer()
        print("--- Composite: %s seconds ---" % (comp_end - comp_start))
        print("---------------------")
    # save and return
    resulting_image = cupy.asnumpy(resulting_image_gpu)
    cv2.imwrite(result_path.__str__(), resulting_image)
    return result_path


def cpu_motion_blur(video_id, blur_strength, blur_transparency, frame_skip, generate_blur):
    return Path(get_motion_blur_image(video_id, "motion_blur.png"))


def gpu_blur_image(image, blur_strength, blur_transparency, frame_skip, angle, magnitude, rios, center, show):
    # create kernel based on blur_strength, magnitude and frame_skip
    kernel_size = int(magnitude * 2 + blur_strength * 3 + frame_skip * 2)
    kernel = cupy.zeros((1, kernel_size), dtype=cupy.float32)
    kernel[0] = 1.0 / kernel_size

    if show:
        print(kernel_size)
        print(kernel)

    # Crop ROI
    cropped = image[rios[1]:rios[3], rios[0]:rios[2]]
    center_y = center[1]
    center_x = center[0]

    if show:
        show_image = cupy.asnumpy(cropped)
        plt.imshow(show_image)
        ax = plt.gca()
        ax.set_facecolor("black")
        plt.show()

    # Rotate ROI
    rotated = cupy.asarray(scipy.ndimage.rotate(cropped.get(), angle.get(), reshape=True, mode='constant', cval=0))
    if show:
        show_image = cupy.asnumpy(rotated)
        plt.imshow(show_image)
        ax = plt.gca()
        ax.set_facecolor("black")
        plt.show()

    # Stretch and blur
    stretched = cupyx.scipy.ndimage.zoom(rotated, (1, magnitude, 1), order=1)
    if show:
        show_image = cupy.asnumpy(stretched)
        plt.imshow(show_image)
        ax = plt.gca()
        ax.set_facecolor("black")
        plt.show()

    result = cupy.zeros_like(stretched)
    for channel in range(4):  # Loop over RGBA channels
        result[:, :, channel] = cupyx.scipy.ndimage.convolve(stretched[:, :, channel], kernel, mode='constant', cval=0)
    if show:
        show_image = cupy.asnumpy(result)
        plt.imshow(show_image)
        ax = plt.gca()
        ax.set_facecolor("black")
        plt.show()
    # Rotate back
    derotated = cupy.asarray(scipy.ndimage.rotate(result.get(), -angle.get(), reshape=True, mode='constant', cval=0))

    if show:
        show_image = cupy.asnumpy(derotated)
        plt.imshow(show_image)
        ax = plt.gca()
        ax.set_facecolor("black")
        plt.show()
    # Ensure sizes fit
    derotated_height, derotated_width = derotated.shape[0], derotated.shape[1]
    new_y1 = center_y - derotated_height // 2
    new_y2 = new_y1 + derotated_height
    new_x1 = center_x - derotated_width // 2
    new_x2 = new_x1 + derotated_width
    if new_y1 >= new_y2 or new_x1 >= new_x2:
        raise ValueError("Invalid coordinates: new_y1 >= new_y2 or new_x1 >= new_x2")

    crop_y1 = max(0, -new_y1)
    crop_y2 = derotated_height - max(0, new_y2 - image.shape[0])
    crop_x1 = max(0, -new_x1)
    crop_x2 = derotated_width - max(0, new_x2 - image.shape[1])

    derotated_cropped = derotated[crop_y1:crop_y2, crop_x1:crop_x2]

    new_y1 = max(0, new_y1)
    new_y2 = min(image.shape[0], new_y2)
    new_x1 = max(0, new_x1)
    new_x2 = min(image.shape[1], new_x2)

    if show:
        show_image = cupy.asnumpy(derotated_cropped)
        plt.imshow(show_image)
        ax = plt.gca()
        ax.set_facecolor("black")
        plt.show()


    # Put back in original frame
    image[new_y1:new_y2, new_x1:new_x2] = derotated_cropped
    image[:, :, 3] = (image[:, :, 3] * blur_transparency).astype(np.uint8)

    if show:
        show_image = cupy.asnumpy(image)
        plt.imshow(show_image)
        ax = plt.gca()
        ax.set_facecolor("black")
        plt.show()

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

        blurred_frame = create_blurred_frame_global_kernel(to_be_blurred_frame, magnitude, angle, min_x, min_y, max_x,max_y)
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


def get_max_min_center_of_object_gpu(frame):
    alpha_channel = frame[..., 3]
    visible_pixels = cupy.argwhere(alpha_channel > 0)
    min_y, min_x = cupy.min(visible_pixels, axis=0).get()
    max_y, max_x = cupy.max(visible_pixels, axis=0).get()
    min_y = min_y if min_y < 1 else min_y -1
    min_x = min_x if min_x < 1 else min_x -1
    min_x = min_x if min_x == frame.shape[0] else min_x + 1
    max_y = max_y if max_y == frame.shape[1] else max_y + 1
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

        # Alle Frames laden
        foreground_frames = sorted(Path(foreground_folder).glob("*.png"))
        background_frames = sorted(Path(background_folder).glob("*.png"))
        if not foreground_frames or not background_frames:
            raise ValueError("No frames found in the specified folders.")

        # Berechnung des Startindexes basierend auf dem negativen Offset
        start_frame_index = len(foreground_frames) - 1 + frame_offset
        if start_frame_index < 0 or start_frame_index >= len(foreground_frames):
            raise ValueError("Frame offset is out of range.")

        # Hintergrund basierend auf `start_frame_index` laden
        background_frame_path = str(background_frames[start_frame_index])
        background_frame = cv2.imread(background_frame_path, cv2.IMREAD_UNCHANGED)
        background_frame = cv2.cvtColor(background_frame, cv2.COLOR_BGR2BGRA)

        # Begrenze die Anzahl der Instanzen basierend auf den verfügbaren Frames
        instance_count = instance_count - 1
        max_instances = (start_frame_index + 1) // frame_skip
        instance_count = min(instance_count, max_instances)

        for i in range(instance_count):
            # Berechnung des Frame-Indexes für den Vordergrund
            frame_index = -(instance_count * frame_skip) + (i * frame_skip) + start_frame_index
            if frame_index < 0 or frame_index >= len(foreground_frames):
                continue  # Überspringe ungültige Indizes

            foreground_frame = cv2.imread(str(foreground_frames[frame_index]), cv2.IMREAD_UNCHANGED)
            if foreground_frame is None:
                continue

            # Transparenzberechnung
            if transparency_mode == "uniform":
                alpha = int(255 * transparency_strength)
            elif transparency_mode == "gradient linear":
                alpha = int(255 * ((i + 1) / instance_count) * transparency_strength)
                alpha = max(1, min(255, alpha))  # Sicherstellen, dass Alpha mindestens 1 ist
            elif transparency_mode == "gradient quadratic":
                alpha = int(255 * ((i + 1) / instance_count) ** 0.5 * transparency_strength)

            # Alpha-Kanal der Maske anpassen
            foreground_frame[:, :, 3] = (foreground_frame[:, :, 3] * (alpha / 255)).astype(np.uint8)

            # Maske erstellen und Alpha-Kanal separat behandeln
            mask = foreground_frame[:, :, 3] > 0
            foreground_alpha = foreground_frame[:, :, 3]
            # Alpha-Blending anwenden
            for c in range(3):  # Nur RGB-Kanäle
                background_frame[:, :, c][mask] = (
                        foreground_frame[:, :, c][mask] * (foreground_alpha[mask] / 255) +
                        background_frame[:, :, c][mask] * (1 - foreground_alpha[mask] / 255)
                ).astype(np.uint8)

            # Alpha-Kanal aktualisieren (maximieren für additive Transparenz)
            background_frame[:, :, 3][mask] = np.maximum(background_frame[:, :, 3][mask], foreground_alpha[mask])

        # Verschobenen letzten Vordergrund-Frame hinzufügen
        last_foreground_frame_path = str(foreground_frames[start_frame_index])
        last_foreground_frame = cv2.imread(last_foreground_frame_path, cv2.IMREAD_UNCHANGED)

        # Bereinigen der Segmentierungsmaske des letzten Frames
        alpha_channel = last_foreground_frame[:, :, 3]
        mask = alpha_channel > 0

        # Entfernen des Alphakanals aus dem letzten Vordergrund-Frame
        for c in range(3):
            background_frame[:, :, c][mask] = last_foreground_frame[:, :, c][mask]

        # Sicherstellen, dass der Alphakanal korrekt ist (für nicht maskierte Bereiche)
        background_frame[:, :, 3][mask] = 255

        # Speichern des resultierenden Bildes
        output_image = cv2.cvtColor(background_frame, cv2.COLOR_BGRA2RGBA)
        output_image = Image.fromarray(output_image)
        output_image.save(output_path, format="PNG")
        print(f"Multiple instance effect created with offset {frame_offset} and saved to {output_path}")

    except Exception as e:
        print(f"Error creating multiple instance effect: {e}")


def create_multiple_instance_effect_reversed(
        video_id, output_path, instance_count, frame_skip, transparency_mode="uniform", transparency_strength=0.5,
        frame_offset=0
):
    try:
        foreground_folder = get_foreground_temp_image_folder(video_id)
        background_folder = get_background_temp_image_folder(video_id)

        # Alle Hintergrund- und Vordergrundframes laden
        background_frames = sorted(Path(background_folder).glob("*.png"))
        foreground_frames = sorted(Path(foreground_folder).glob("*.png"))

        # Berechne das maximale Offset, das möglich ist
        max_offset = len(background_frames) - 1
        frame_offset = min(frame_offset, max_offset)

        # Passe den Hintergrund basierend auf dem Offset an
        background_frame_path = str(
            background_frames[frame_offset])  # Hintergrund-Frame entsprechend dem Offset auswählen
        background_frame = cv2.imread(background_frame_path, cv2.IMREAD_UNCHANGED)
        background_frame = cv2.cvtColor(background_frame, cv2.COLOR_BGR2BGRA)

        # Passe die Vordergrundframes basierend auf dem Offset an
        foreground_frames = foreground_frames[frame_offset:]  # Frames entsprechend dem Offset reduzieren

        # Aktualisiere die maximale Anzahl der Instanzen
        instance_count = instance_count - 1
        instance_count = min(instance_count, len(foreground_frames) // frame_skip)

        # Frames durchlaufen
        for i in range(instance_count, -1, -1):
            frame_index = i * frame_skip
            if frame_index >= len(foreground_frames):
                continue  # Überspringe, falls der Index außerhalb des Bereichs liegt

            foreground_frame = cv2.imread(str(foreground_frames[frame_index]), cv2.IMREAD_UNCHANGED)

            # Transparenz für den ersten Frame auf volle Sichtbarkeit setzen
            if i == 0:
                alpha = 255
            else:
                # Transparenzberechnung für alle anderen Frames
                if transparency_mode == "uniform":
                    alpha = int(255 * transparency_strength)
                elif transparency_mode == "gradient linear":
                    alpha = int(255 * (1 - (i / instance_count)) * transparency_strength)
                    alpha = max(1, min(255, alpha))  # Sicherstellen, dass Alpha mindestens 1 ist
                elif transparency_mode == "gradient quadratic":
                    offset = 0.7  # Steuert, wie sanft der Übergang startet
                    progress = (i / instance_count) + offset
                    alpha = int(255 * (1 - (progress ** 0.5 - offset ** 0.5)) * transparency_strength)
                    alpha = max(1, min(255, alpha))  # Sicherstellen, dass Alpha mindestens 1 ist

            print(f"Mode: {transparency_mode}, Alpha Value for instance {i + 1}/{instance_count + 1}: {alpha}")

            # Alpha-Kanal der Maske anpassen
            foreground_frame[:, :, 3] = (foreground_frame[:, :, 3] * (alpha / 255)).astype(np.uint8)

            # Debugging: Überprüfen der Alpha-Werte
            print(f"Adjusted Alpha (Instance {i + 1}): {np.unique(foreground_frame[:, :, 3])}")

            # Maske erstellen und Alpha-Kanal separat behandeln
            mask = foreground_frame[:, :, 3] > 0
            foreground_alpha = foreground_frame[:, :, 3]
            # Alpha-Blending anwenden
            for c in range(3):  # Nur RGB-Kanäle
                background_frame[:, :, c][mask] = (
                        foreground_frame[:, :, c][mask] * (foreground_alpha[mask] / 255) +
                        background_frame[:, :, c][mask] * (1 - foreground_alpha[mask] / 255)
                ).astype(np.uint8)

            # Alpha-Kanal aktualisieren (maximieren für additive Transparenz)
            background_frame[:, :, 3][mask] = np.maximum(background_frame[:, :, 3][mask], foreground_alpha[mask])

        # Speichern des resultierenden Bildes
        output_image = cv2.cvtColor(background_frame, cv2.COLOR_BGRA2RGBA)
        output_image = Image.fromarray(output_image)
        output_image.save(output_path, format="PNG")
        print(f"Multiple instance effect (reversed) with offset {frame_offset} created and saved to {output_path}")

    except Exception as e:
        print(f"Error creating reversed multiple instance effect: {e}")


def create_multiple_instance_effect_middle(
        video_id, output_path, instance_count, frame_skip, transparency_mode="uniform", transparency_strength=0.5,
        frame_offset=0
):
    try:
        foreground_folder = get_foreground_temp_image_folder(video_id)
        background_folder = get_background_temp_image_folder(video_id)

        # Alle Vordergrund- und Hintergrundframes laden
        foreground_frames = sorted(Path(foreground_folder).glob("*.png"))
        background_frames = sorted(Path(background_folder).glob("*.png"))

        # Berechne den Referenzindex basierend auf dem Offset
        middle_index = len(foreground_frames) // 2
        reference_index = middle_index + frame_offset
        reference_index = max(0, min(reference_index, len(foreground_frames) - 1))  # Begrenzen auf gültigen Bereich

        # Hintergrund initialisieren
        background_frame_path = str(background_frames[reference_index])
        background_frame = cv2.imread(background_frame_path, cv2.IMREAD_UNCHANGED)
        background_frame = cv2.cvtColor(background_frame, cv2.COLOR_BGR2BGRA)

        # Symmetrische Auswahl der Frames basierend auf dem Referenzframe und frame_skip
        left_indices = list(range(reference_index - frame_skip, -1, -frame_skip))  # Links vom Referenzframe
        right_indices = list(
            range(reference_index + frame_skip, len(foreground_frames), frame_skip))  # Rechts vom Referenzframe

        # Kombiniere beide Seiten und sortiere nach Entfernung vom Referenzframe
        selected_indices = left_indices + right_indices
        selected_indices.sort(key=lambda x: abs(x - reference_index))

        # Beschränke die Anzahl der ausgewählten Frames basierend auf instance_count
        selected_indices = selected_indices[:instance_count - 1]

        # Frames entsprechend der Reihenfolge
        sorted_frames = [foreground_frames[i] for i in selected_indices]

        # Füge die Frames ein
        for frame_path in sorted_frames:
            foreground_frame = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)

            # Distanz zum Referenzframe
            frame_index = foreground_frames.index(frame_path)
            distance_from_reference = abs(frame_index - reference_index)
            max_distance = max(reference_index, len(foreground_frames) - reference_index)

            # Transparenzberechnung
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

            # Alpha-Kanal anpassen
            foreground_frame[:, :, 3] = (foreground_frame[:, :, 3] * (alpha / 255)).astype(np.uint8)

            # Maske erstellen und Alpha-Blending anwenden
            mask = foreground_frame[:, :, 3] > 0
            foreground_alpha = foreground_frame[:, :, 3]

            # Hintergrund und Vordergrund mischen
            for c in range(3):  # Nur RGB-Kanäle
                background_frame[:, :, c] = np.where(
                    mask,
                    foreground_frame[:, :, c] * (foreground_alpha / 255) +
                    background_frame[:, :, c] * (1 - foreground_alpha / 255),
                    background_frame[:, :, c]
                ).astype(np.uint8)

            # Alpha-Kanal aktualisieren
            background_frame[:, :, 3] = np.where(
                mask,
                np.maximum(background_frame[:, :, 3], foreground_alpha),
                background_frame[:, :, 3]
            ).astype(np.uint8)

        # Zum Schluss den Referenzframe als Vordergrund einfügen
        reference_frame = cv2.imread(str(foreground_frames[reference_index]), cv2.IMREAD_UNCHANGED)
        mask = reference_frame[:, :, 3] > 0
        for c in range(3):  # RGB-Kanäle
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

        # Ergebnis speichern
        output_image = cv2.cvtColor(background_frame, cv2.COLOR_BGRA2RGBA)
        output_image = Image.fromarray(output_image)
        output_image.save(output_path, format="PNG")
        print(f"Multiple instance effect with reference offset {frame_offset} created and saved to {output_path}")

    except Exception as e:
        print(f"Error creating multiple instance effect: {e}")
