import os
import traceback
from typing import Annotated


import static_ffmpeg
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from timeit import default_timer as timer


os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from video_editing import get_video_details
from video_editing import save_video
from video_editing import initialize_segmentation
from video_editing import add_new_point_to_segmentation
from video_editing import get_masked_video
from video_editing import cut_video
from video_editing import get_frame
from path_manager import create_all_paths, get_multiple_instances_image, get_motion_blur_image, delete_project, \
    get_upload_path
from image_editing import create_multiple_instance_effect, create_multiple_instance_effect_reversed
from image_editing import create_multiple_instance_effect_middle, create_motion_blur_image
from image_editing import save_background
from image_effects import process_effect_request


from project_data import get_all_projects, get_background_type
from project_data import get_step_data
from project_data import create_project
from project_data import set_current_step
from project_data import Step
from project_data import BackgroundType, set_background_type


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

@app.get("/get-video")
async def get_video(video_id: str):
    video_path = get_upload_path(video_id)
    return FileResponse(video_path, media_type="video/mp4")

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

@app.post("/set-background-type")
async def set_current_background_type(video_id: str = Form(...), background_type: str = Form(...)):
    if background_type == "custom":
        background = BackgroundType.CUSTOM
    elif background_type == "transparent":
        background = BackgroundType.TRANSPARENT
    elif background_type == "video_frame":
        background = BackgroundType.VIDEO_FRAME
    else:
        return JSONResponse(
            status_code=500,
            content={"message": "Failed to set background type", "error": background_type + "is not a valid background_type."},
        )
    set_background_type(video_id, background)
    return JSONResponse(status_code=200, content=video_id)


@app.get("/get-background-type")
async def get_current_background_type(video_id):
    try:
        background_type = get_background_type(video_id)
        return JSONResponse(status_code=200, content=background_type)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": "Failed to access background type from video", "error": str(e)},
        )


@app.get("/get-motion-blur-preview")
async def get_motion_blur_preview(video_id: str, blur_strength: float , blur_transparency: float, frame_skip: int):
    try:
        print("Generating motion blur preview with ", blur_strength, blur_transparency, frame_skip)
        start = timer()
        image_path = await create_motion_blur_image(video_id, blur_strength, blur_transparency, frame_skip)
        end = timer()
        print("--- Total Generation Time: %s seconds ---" % (end - start))
        print("-----------------------------------------------------------")

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
        details = await get_video_details(video_id)
        return JSONResponse(status_code=200, content=details)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": "Failed to get video details", "error": str(e)},
        )

@app.get("/total-frame-count")
async def total_frame_count(video_id: str):
    try:
        details = await get_video_details(video_id)
        return JSONResponse(status_code=200, content=details["total_frames"])
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": "Failed to get video details", "error": str(e)},
        )

@app.get("/initialize-segmentation")
async def get_first_frame_of_video(video_id: str):
    try:
        await initialize_segmentation(video_id)
        print("Successfully initialized segmentation")
        set_current_step(video_id, Step.SEGMENTATION)
        return JSONResponse(status_code=200, content="")
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": "Failed to get first frame", "error": str(e)},
        )

@app.post("/add-point")
async def add_point_to_video(video_id: Annotated[str, Form()], point_x: Annotated[float, Form()], point_y: Annotated[float, Form()], point_type: Annotated[int, Form()], frame_num: Annotated[int, Form()]):
    try:
        masked_frame = await add_new_point_to_segmentation(video_id, point_x, point_y, point_type, frame_num)
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

    try:
        print(video_id, start_time, end_time)

        cut_video_path = await cut_video(video_id, start_time, end_time)
        print("Created cut_video at path:" + cut_video_path.__str__())

        return FileResponse(cut_video_path, media_type="video/mp4")

    except Exception as e:
        print(e)
        print(e.__traceback__)
        print(traceback.format_exc())

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

@app.delete("/delete-video")
async def delete_video(video_id: str):

    try:
        if delete_project(video_id):
            return JSONResponse(status_code=200, content={"message": f"Projekt {video_id} wurde erfolgreich gelöscht"})
        else:
            return JSONResponse(status_code=404, content={"message": "Projekt nicht gefunden oder konnte nicht gelöscht werden"})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Fehler beim Löschen des Projekts {video_id}", "error": str(e)}
        )


@app.get("/get-frame")
async def get_first_frame_of_video(video_id: str, frame_num: int):
    try:
        first_frame_path = await get_frame(video_id, frame_num)
        print("Got frame: " + first_frame_path.__str__())
        return FileResponse(first_frame_path, media_type="image/jpeg")
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": "Failed to get first frame", "error": str(e)},
        )


@app.get("/effect/multiple-instances/")
async def multiple_instance_effect(
    video_id: str,
    instance_count: int,
    frame_skip: int,
    transparency_mode: str = "uniform",
    transparency_strength: float = 0.5,
    frame_reference: str = "last",  # "first", "middle" oder "last"
    frame_offset: int = 0  # Optionaler Offset für den Referenzframe
):
    try:
        create_all_paths(video_id)
        output_path = get_multiple_instances_image(video_id, "multiple_instances_result.png")

        # Wähle die Methode basierend auf frame_reference
        if frame_reference == "middle":
            create_multiple_instance_effect_middle(
                video_id,
                str(output_path),
                instance_count,
                frame_skip,
                transparency_mode,
                transparency_strength,
                frame_offset
            )
        elif frame_reference == "first":
            create_multiple_instance_effect_reversed(
                video_id,
                str(output_path),
                instance_count,
                frame_skip,
                transparency_mode,
                transparency_strength,
                frame_offset
            )
        else:  # "last"
            create_multiple_instance_effect(
                video_id,
                str(output_path),
                instance_count,
                frame_skip,
                transparency_mode,
                transparency_strength,
                frame_offset
            )

        if not output_path.exists():
            raise FileNotFoundError(f"Das Bild wurde nicht unter {output_path} gespeichert.")

        return FileResponse(output_path, media_type="image/png")
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/final-effects-preview")
async def get_final_effects_preview(video_id: str, effect_type: str):

    try:
        set_current_step(video_id, Step.AFTER_EFFECT)
        if effect_type == "motion_blur":
            image_path = get_motion_blur_image(video_id, "motion_blur.png")
        elif effect_type == "multiple_instances":
            image_path = get_multiple_instances_image(video_id, "multiple_instances_result.png")
        else:
            raise ValueError("Invalid effect type provided.")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image for effect type {effect_type} not found.")

        return FileResponse(image_path, media_type="image/png")
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Failed to load final effects preview: {str(e)}"},
        )

@app.post("/apply-final-effects")
async def apply_final_effects(video_id: Annotated[str, Form()], effect_type: Annotated[str, Form()], brightness: Annotated[float, Form()],
                              contrast: Annotated[float, Form()], saturation: Annotated[float, Form()]):

    try:
        if effect_type == "motion-blur":
            image_path = get_motion_blur_image(video_id, "motion_blur.png")
        else:
            image_path = get_multiple_instances_image(video_id, "multiple_instances_result.png")
        set_current_step(video_id, Step.AFTER_EFFECT)
        output_image_path = process_effect_request(video_id, brightness, contrast, saturation, image_path)
        return FileResponse(output_image_path, media_type="image/png")
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": "Failed to apply final effects", "error": str(e)}
        )