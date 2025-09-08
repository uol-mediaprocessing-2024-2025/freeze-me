import static_ffmpeg
static_ffmpeg.add_paths()
import ffmpeg
import pathlib
import os

def extract_first_frames():
    in_dir = pathlib.Path("video_candidates")
    out_dir = pathlib.Path("images/part_two/first_frame")
    print(f"Extracting all first frames from {in_dir} and saving to {out_dir}")
    categories = ['billiard', 'football', 'gymnastics', 'rocket_league', 'street_traffic']

    for file in os.listdir(in_dir):
        if file.endswith(".mp4"):
            file_path = os.path.join(in_dir, file)
            new_file_name = pathlib.Path(file).with_suffix("").name
            output_path = os.path.join(out_dir, f"{new_file_name}.jpg")

            (
                ffmpeg
                .input(str(file_path))
                .output(str(output_path), **{"frames:v": 1, "q:v": 2})
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )

if __name__ == "__main__":
    extract_first_frames()