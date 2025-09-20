import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, List, Dict

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.responses import FileResponse

app = FastAPI()

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (adjust for production use)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

results_path = Path("results")

effects = ["action_line", "first_frame", "key_frame", "motion_blur", "multiple_instances"]
videos_png = ["billiard-13.png", "football-12.png", "gymnastics-9.png", "rocket-league-11.png", "street-traffic-13.png"]
videos_jpg = ["billiard-13.jpg", "football-12.jpg", "gymnastics-9.jpg", "rocket-league-11.jpg", "street-traffic-13.jpg"]
part_two_categories = ["billiard", "football", "gymnastics", "rocket-league", "street-traffic"]
part_two_choice = ["left", "center", "right"]

# Williams (Balanced Latin) Square für n=5
WILLIAMS_5 = [
    [1, 5, 2, 4, 3],
    [2, 1, 3, 5, 4],
    [3, 2, 4, 1, 5],
    [4, 3, 5, 2, 1],
    [5, 4, 1, 3, 2],
]

WILLIAMS_3 = [
    [1, 2, 3],
    [2, 3, 1],
    [3, 1, 2],
    [3, 2, 1],
    [2, 1, 3],
    [1, 3, 2],
]


def part_one_for_participant(
        p: int
) -> List[Dict]:
    n = 5
    p = p % 10

    base_row = (p - 1) % n
    seq = WILLIAMS_5[base_row][:]
    if p > n:
        seq = list(reversed(seq))  # Reverse für Teilnehmer 6..10

    trials = []
    for pos, k in enumerate(seq, start=1):  # Blöcke 0..4
        eff_label = effects[k - 1]
        video_index = ((k + ((p - 1) % n) - 1) % n) + 1  # 1..5
        if eff_label == "key_frame" or eff_label == "first_frame":
            vid_label = videos_jpg[video_index - 1]
        else:
            vid_label = videos_png[video_index - 1]
        trials.append({
            "participant": p,
            "order": pos,
            "effect_index": k,
            "effect": eff_label,
            "video_index": video_index,
            "video": vid_label,
        })
    return trials


def part_two(p: int) -> List[Dict]:
    n = 5
    p = p % 10  # deine bisherige Normalisierung (1..10)

    # --- 5er-Faktor (wie bei dir) ---
    base_row = (p - 1) % n
    seq = WILLIAMS_5[base_row][:]  # k ∈ {1..5}
    if p > n:
        seq = list(reversed(seq))  # Teilnehmer 6..10 gespiegelt

    # --- 3er-Faktor vorbereiten ---
    # Wähle eine der 6 Williams-3-Sequenzen abhängig vom Teilnehmer,
    # dann auf 5 Trials wiederholen/abschneiden.
    w3_row = (p - 1) % 6
    seq3 = WILLIAMS_3[w3_row][:]
    # Auf 5 Trials bringen (tile & cut), anschließend in {0,1,2} mappen
    seq3_tiled = (seq3 * ((5 + len(seq3) - 1) // len(seq3)))[:5]
    seq3_zero_based = [x - 1 for x in seq3_tiled]  # -> {0,1,2}

    trials = []
    for pos, (k, choice_val) in enumerate(zip(seq, seq3_zero_based), start=1):  # pos 1..5
        eff_label = effects[k - 1]
        video_index = ((k + ((p - 1) % n) - 1) % n) + 1  # 1..5

        vid_label = part_two_categories[video_index - 1]

        trials.append({
            "participant": p,
            "order": pos,
            "effect_index": k,
            "effect": eff_label,
            "video_index": video_index,
            "video": vid_label,
            "choice": choice_val,  # neu: 0, 1 oder 2
        })

    return trials


def create_random_folder(
        parent_dir: os.PathLike,
        *,
        prefix: str = "",
        ensure_parents: bool = True,
        name_len: int = 12,
        retries: int = 5,
) -> Path:
    parent = Path(parent_dir)
    if ensure_parents:
        parent.mkdir(parents=True, exist_ok=True)
    elif not parent.exists():
        raise FileNotFoundError(f"Parent directory does not exist: {parent}")

    for _ in range(max(1, retries)):
        rnd = uuid.uuid4().hex[:max(1, name_len)]
        name = f"{prefix}{rnd}"
        target = parent / name
        try:
            target.mkdir(mode=0o755, exist_ok=False)
            return target
        except FileExistsError:
            continue

    raise FileExistsError(f"Could not create a unique directory in {parent} after {retries} attempts.")


def read_json(file_path: os.PathLike, default: Any = None, *, encoding: str = "utf-8") -> Any:
    p = Path(file_path)
    try:
        text = p.read_text(encoding=encoding)
    except FileNotFoundError:
        return default

    text = text.strip()
    if not text:
        return default

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Ungültige JSON-Datei: {p} ({e})") from e


def write_json(
        file_path: os.PathLike,
        data: Any,
        *,
        encoding: str = "utf-8",
        mkdirs: bool = True,
        atomic: bool = True,
        indent: int = 2,
) -> Path:
    p = Path(file_path)
    if mkdirs:
        p.parent.mkdir(parents=True, exist_ok=True)

    serialized = json.dumps(data, ensure_ascii=False, indent=indent)

    if atomic:
        fd, tmp_path = tempfile.mkstemp(prefix="._tmp-", dir=str(p.parent))
        try:
            with os.fdopen(fd, "w", encoding=encoding) as tmpf:
                tmpf.write(serialized)
                tmpf.flush()
                os.fsync(tmpf.fileno())
            os.replace(tmp_path, p)  # atomarer Swap (POSIX & Windows)
        except Exception:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise
    else:
        p.write_text(serialized, encoding=encoding)

    return p


@app.post("/create_user_folder")
async def upload_video():
    try:
        path = create_random_folder(results_path).name.__str__()
        return JSONResponse(status_code=200, content=path)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": "Failed to create user folder", "error": str(e)},
        )


@app.get("/send_demographic")
async def send_demographic_answers(user_path: str, age: str, color_blind: str, education: str, frequency: str):
    try:
        json_data = {
            "demographics": {
                "age": age,
                "color_blind": color_blind,
                "education": education,
                "frequency": frequency
            },
            "part_one": [],
            "part_two": []
        }
        write_json(Path(os.path.join("results", user_path, "answers.json")), json_data)
        return JSONResponse(status_code=200, content=json_data)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": "Failed to create save demographics", "error": str(e)},
        )


@app.get("/get_image_paths")
async def get_image_paths():
    try:
        user_count = len(os.listdir("results"))
        result = part_one_for_participant(user_count)
        image_paths = []
        for result in result:
            image_path = os.path.join("images", "part_one", result["effect"], result["video"])
            print(image_path)
            image_paths.append(image_path)

        return JSONResponse(status_code=200, content=image_paths)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": "Failed to create save demographics", "error": str(e)},
        )


@app.get("/get_part_two_data")
async def get_image_paths_two():
    try:
        user_count = len(os.listdir("results"))
        result = part_two(user_count)
        data = []
        count = 1
        for result in result:
            base_path = os.path.join("images", "part_two", result["effect"], result["video"])
            file_list = os.listdir(base_path)
            image_paths = []
            video_name = ""
            for i in range(len(file_list)):
                file = file_list[i]
                if i == result["choice"]:
                    video_name = Path(file).stem + ".mp4"
                image_path = os.path.join(base_path, file)
                print(image_path)
                image_paths.append(os.path.abspath(image_path))
            video_path = os.path.abspath(os.path.join("video_candidates", video_name))
            print(video_path)
            thumbnail = os.path.abspath(os.path.join("thumbnails", count.__str__() + ".png"))
            all_data = {
                "image_paths": image_paths,
                "video_path": video_path,
                "choice": result["choice"],
                "thumbnail": thumbnail,
            }
            data.append(all_data)
            count += 1

        return JSONResponse(status_code=200, content=data)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": "Failed to create get image paths", "error": str(e)},
        )


@app.get("/get_image")
async def get_image(path: str):
    try:
        media_type = "image/png" if Path(path).suffix == ".png" else "image/jpeg"
        output_path = os.path.abspath(path)
        print(output_path)
        return FileResponse(output_path, media_type=media_type)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": "Failed to create get image", "error": str(e)},
        )


@app.get("/get_video")
async def get_video(path: str):
    try:
        media_type = "video/mp4"
        output_path = os.path.abspath(path)
        print(output_path)
        return FileResponse(output_path, media_type=media_type)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": "Failed to create get video", "error": str(e)},
        )


@app.get("/send_part_one_answer")
async def send_part_one_answer(user_path: str, image_path: str, answer: str):
    try:
        json_data = read_json(Path(os.path.join("results", user_path, "answers.json")))
        answer = {
            "image_path": image_path,
            "answer": answer
        }
        json_data["part_one"].append(answer)
        write_json(Path(os.path.join("results", user_path, "answers.json")), json_data)
        return JSONResponse(status_code=200, content=answer)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": "Failed to create save answer", "error": str(e)},
        )


@app.get("/send_part_two_answer")
async def send_part_two_answer(user_path: str, video_path: str, answer: int, answer_path: str, correct: int, correct_path: str):
    try:
        json_data = read_json(Path(os.path.join("results", user_path, "answers.json")))
        answer = {
            "video_path": video_path,
            "answer": answer,
            "answer_path": answer_path,
            "correct": correct,
            "correct_path": correct_path
        }
        json_data["part_two"].append(answer)
        write_json(Path(os.path.join("results", user_path, "answers.json")), json_data)
        return JSONResponse(status_code=200, content=answer)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": "Failed to create save answer", "error": str(e)},
        )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001)
