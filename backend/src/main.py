from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from image_processing import process_image_blur
import io
import uvicorn

app = FastAPI()

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (adjust for production use)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)


@app.post("/apply-blur")
async def apply_blur(file: UploadFile = File(...)):
    """
    API endpoint to apply a blur effect to an uploaded image.
    :param file: The uploaded image file.
    :return: StreamingResponse containing the blurred image.
    """
    try:
        # Process the image
        blurred_image = await process_image_blur(file)

        # Create a BytesIO object to send the image as a stream
        image_io = io.BytesIO()
        blurred_image.save(image_io, format="PNG")
        image_io.seek(0)

        # Return the blurred image as a stream in the response
        return StreamingResponse(image_io, media_type="image/png")
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": "Failed to process image", "error": str(e)},
        )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
