import os

import static_ffmpeg
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import numpy as np
import matplotlib.pyplot as plt

from video_editing import get_video_details

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

@app.get("/cut-object")
async def cut_out_object_from_video(path):
    try:
        result = await cut_out_object_from_video(path)
        return JSONResponse(status_code=200, content=result)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content= {"message": "Failed to cut out object from video.", "error": str(e)},
        )



@app.post("/video-details")
async def video_details(file: UploadFile = File(...)):
    try:
        details = await get_video_details(file)
        return JSONResponse(status_code=200, content=details)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": "Failed to get video details", "error": str(e)},
        )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
