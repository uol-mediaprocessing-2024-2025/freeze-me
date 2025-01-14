import io
import os
import traceback
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
from video_editing import cut_video
from video_editing import get_frame
from path_manager import create_all_paths, get_multiple_instances_image
from image_editing import create_multiple_instance_effect
from image_editing import save_background
from image_editing import generate_motion_blur_image

from project_data import get_all_projects
from project_data import get_step_data
from project_data import create_project
from project_data import set_current_step
from project_data import Step

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
        create_project(video_id)
        return JSONResponse(status_code=200, content=video_id)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": "Failed to get upload video", "error": str(e)},
        )

@app.post("/upload-background")
async def upload_background(file: UploadFile = File(...), video_id: str = Form(...)):
    try:
        background_path = await save_background(file, video_id)
        return FileResponse(background_path, media_type="image/jpeg")
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": "Failed to upload background", "error": str(e)},
        )

@app.get("/get-motion-blur-preview")
async def get_motion_blur_preview(video_id: str, blur_strength: float , blur_transparency: float, frame_skip: int):
    try:
        print("Generating motion blur preview with ", blur_strength, blur_transparency, frame_skip)
        image_path = await generate_motion_blur_image(video_id, blur_strength, blur_transparency, frame_skip)
        return FileResponse(image_path, media_type="image/png")
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"message": "Failed to create motion blur preview", "error": str(e)},
        )


@app.get("/video-details")
async def video_details(video_id: str):
    try:
        print("Getting details from: " + video_id)
        set_current_step(video_id, Step.SEGMENTATION)
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
        first_frame_path = await get_first_frame(video_id)
        print("Got first frame: " + first_frame_path.__str__())
        return FileResponse(first_frame_path, media_type="image/jpeg")
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": "Failed to get first frame", "error": str(e)},
        )

@app.post("/add-point")
async def add_point_to_video(video_id: Annotated[str, Form()], point_x: Annotated[float, Form()], point_y: Annotated[float, Form()], point_type: Annotated[int, Form()]):
    try:
        masked_frame = await add_new_point_to_segmentation(video_id, point_x, point_y, point_type)
        return FileResponse(masked_frame, media_type="image/png")
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": "Failed to get first frame", "error": str(e)},
        )

@app.get("/get-segmentation-result")
async def get_segmentation_result(video_id):
    try:
        masked_video = await get_masked_video(video_id)
        set_current_step(video_id, Step.BACKGROUND)
        set_current_step(video_id, Step.MAIN_EFFECT)
        return FileResponse(masked_video, media_type="video/mp4")
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": "Failed to get first frame", "error": str(e)},
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


@app.post("/cut-video")
async def cut_video_endpoint(video_id: Annotated[str, Form()], start_time: Annotated[float, Form()], end_time: Annotated[float, Form()]):
    """
    API-Endpunkt, um ein Video auf einen bestimmten Zeitraum zu schneiden.
    """
    try:
        # Hier rufen wir die cut_video-Methode auf und erhalten den Pfad zum geschnittenen Video
        print(video_id, start_time, end_time)

        cut_video_path = await cut_video(video_id, start_time, end_time)
        print("Created cut_video at path:" + cut_video_path.__str__())
        # Rückgabe der Erfolgsnachricht und des Pfades zum geschnittenen Video
        return FileResponse(cut_video_path, media_type="video/mp4")

    except Exception as e:
        print(e)
        print(e.__traceback__)
        print(traceback.format_exc())
        # Detaillierte Fehlerbehandlung und -meldung
        return JSONResponse(
            status_code=500,
            content={"message": "Fehler beim Schneiden des Videos", "error": str(e)}
        )

@app.get("/video-ids")
async def get_all_video_ids():
    try:
        ids = get_all_projects()
        return JSONResponse(status_code=200, content=ids)

    except Exception as e:
        print(e)
        print(e.__traceback__)
        print(traceback.format_exc())
        # Detaillierte Fehlerbehandlung und -meldung
        return JSONResponse(
            status_code=500,
            content={"message": "Fehler beim Laden der VideoIds", "error": str(e)}
        )

@app.get("/project-data")
async def get_project_progress(video_id: str):
    try:
        data = get_step_data(video_id)
        return JSONResponse(status_code=200, content=data)

    except Exception as e:
        print(e)
        print(e.__traceback__)
        print(traceback.format_exc())
        # Detaillierte Fehlerbehandlung und -meldung
        return JSONResponse(
            status_code=500,
            content={"message": "Fehler beim Laden der VideoIds", "error": str(e)}
        )

@app.get("/get-frame")
async def get_first_frame_of_video(video_id: str):
    try:
        first_frame_path = await get_frame(video_id, 0)
        print("Got first frame: " + first_frame_path.__str__())
        return FileResponse(first_frame_path, media_type="image/jpeg")
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": "Failed to get first frame", "error": str(e)},
        )

@app.get("/effect/multiple-instances/")
async def multiple_instance_effect(video_id: str, instance_count: int, frame_skip: int):
    """
    API-Endpunkt für den Multiple Instance Effect
    """
    try:
        create_all_paths(video_id)  # Sicherstellen, dass alle benötigten Ordner existieren
        output_path = get_multiple_instances_image(video_id, "multiple_instances_result.png")

        # Effekt anwenden
        create_multiple_instance_effect(video_id, str(output_path), instance_count, frame_skip)

        if not output_path.exists():
            raise FileNotFoundError(f"Das Bild wurde nicht unter {output_path} gespeichert.")

        return FileResponse(output_path, media_type="image/png")
    except Exception as e:
        return {"status": "error", "message": str(e)}

