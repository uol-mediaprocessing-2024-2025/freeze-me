import traceback
from pathlib import Path

import cv2
import numpy as np
import os

from matplotlib import pyplot as plt


def adjust_brightness_contrast_saturation(image_path, brightness, contrast, saturation, output_path):

    try:
        # Lade das Bild
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        alpha = image[:, :, 3]
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        if image is None:
            raise FileNotFoundError(f"Image at {image_path} could not be loaded.")

        # Konvertiere das Bild von BGR nach HSV, um Sättigung zu verändern
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv_image[..., 1] *= saturation  # Sättigung anpassen
        hsv_image[..., 1] = np.clip(hsv_image[..., 1], 0, 255)
        hsv_image[..., 2] *= brightness
        hsv_image[..., 2] = np.clip(hsv_image[..., 2], 0, 255)
        image = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # Helligkeit und Kontrast anpassen
        adjusted_image = cv2.convertScaleAbs(image, alpha=contrast)
        adjusted_image = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2BGRA)

        # Copy alpha-channel
        adjusted_image[:, :, 3] = alpha

        # Speichere das Ergebnis
        cv2.imwrite(output_path, adjusted_image)
        return output_path
    except Exception as e:
        print(str(e))
        raise RuntimeError(f"Failed to adjust image properties: {str(e)}")


def process_effect_request(video_id, brightness, contrast, saturation, input_path: Path):

    try:
        output_image_path = os.path.join(input_path.parent.__str__(), input_path.stem + f"_processed.png")

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input image for video ID {video_id} not found.")

        return adjust_brightness_contrast_saturation(
            input_path, brightness, contrast, saturation, output_image_path
        )
    except Exception as e:
        print(e)
        print(e.__traceback__)
        print(traceback.format_exc())
        raise RuntimeError(f"Failed to process effect request: {str(e)}")