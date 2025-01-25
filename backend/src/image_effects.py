import cv2
import numpy as np
import os


def adjust_brightness_contrast_saturation(image_path, brightness, contrast, saturation, output_path):

    try:
        # Lade das Bild
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image at {image_path} could not be loaded.")

        # Konvertiere das Bild von BGR nach HSV, um Sättigung zu verändern
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv_image[..., 1] *= saturation  # Sättigung anpassen
        hsv_image[..., 1] = np.clip(hsv_image[..., 1], 0, 255)
        image = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # Helligkeit und Kontrast anpassen
        adjusted_image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

        # Speichere das Ergebnis
        cv2.imwrite(output_path, adjusted_image)
        return output_path
    except Exception as e:
        raise RuntimeError(f"Failed to adjust image properties: {str(e)}")


def process_effect_request(video_id, brightness, contrast, saturation, output_folder):

    try:
        input_image_path = os.path.join(output_folder, f"{video_id}_input.png")
        output_image_path = os.path.join(output_folder, f"{video_id}_processed.png")

        if not os.path.exists(input_image_path):
            raise FileNotFoundError(f"Input image for video ID {video_id} not found.")

        return adjust_brightness_contrast_saturation(
            input_image_path, brightness, contrast, saturation, output_image_path
        )
    except Exception as e:
        raise RuntimeError(f"Failed to process effect request: {str(e)}")


