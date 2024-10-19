from PIL import Image, ImageFilter
import io


async def process_image_blur(file):
    """
    Applies a blur effect to the uploaded image.
    :param file: The uploaded image file.
    :return: The processed (blurred) PIL Image object.
    """
    # Read the image file data
    image_data = await file.read()

    # Open the image using PIL
    image = Image.open(io.BytesIO(image_data))

    # Apply the blur effect
    blurred_image = image.filter(ImageFilter.BoxBlur(5))

    return blurred_image
