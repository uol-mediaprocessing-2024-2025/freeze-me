import io
import os
from typing import Annotated

import static_ffmpeg
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from starlette.responses import StreamingResponse

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from video_editing import get_video_details
from video_editing import save_video
from video_editing import get_first_frame
from video_editing import add_new_point_to_segmentation
from video_editing import get_masked_video

app = FastAPI()
static_ffmpeg.add_paths()

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (adjust for production use)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)


@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    try:
        video_id = await save_video(file)
        return JSONResponse(status_code=200, content=video_id)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": "Failed to get upload video", "error": str(e)},
        )


@app.get("/video-details")
async def video_details(video_id: str):
    try:
        details = await get_video_details(video_id)
        return JSONResponse(status_code=200, content=details)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": "Failed to get video details", "error": str(e)},
        )

@app.get("/get-first-frame")
async def get_first_frame_of_video(video_id: str):
    try:
        first_frame = await get_first_frame(video_id)
        print("Got first frame")
        return StreamingResponse(first_frame, media_type="image/jpeg")
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": "Failed to get first frame", "error": str(e)},
        )

@app.post("/add-point")
async def add_point_to_video(video_id: Annotated[str, Form()], point_x: Annotated[float, Form()], point_y: Annotated[float, Form()], point_type: Annotated[int, Form()]):
    try:
        masked_frame = await add_new_point_to_segmentation(video_id, point_x, point_y, point_type)
        image_io = io.BytesIO()
        masked_frame.save(image_io, format="PNG")
        image_io.seek(0)
        return StreamingResponse(image_io, media_type="image/png")
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": "Failed to get first frame", "error": str(e)},
        )

@app.get("/get-segmentation-result")
async def get_segmentation_result(video_id):
    try:
        masked_video = await get_masked_video(video_id)
        test = io.FileIO(masked_video)
        return StreamingResponse(test, media_type="video/mp4")
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": "Failed to get first frame", "error": str(e)},
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
