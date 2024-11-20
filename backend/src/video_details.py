import io
from pathlib import Path

import ffmpeg
from fastapi import UploadFile


async def get_video_details(file: UploadFile):
    try:
        path = Path.cwd().joinpath(file.filename)
        video_data = io.BytesIO(await file.read())
        with open(file.filename, "wb") as f:
            f.write(video_data.getbuffer())
        details = ffmpeg.probe(path.__str__(), cmd="static_ffprobe")
        return details
    except Exception as e:
        print(e)
        return ""
