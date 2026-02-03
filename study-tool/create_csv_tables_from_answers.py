import argparse
import csv
import json
from pathlib import Path

PART_ONE_COLUMNS = [
    "Effect", "User (ID)", "Age", "Education", "Frequency", "Color Blind",
    "Video-Category", "Round", "Answer",
    "A1", "A2", "A3", "M1", "M2", "M3", "M4", "L1", "L2", "S1", "S2",
]

PART_TWO_COLUMNS = [
    "Effect", "User (ID)", "Age", "Education", "Frequency", "Color Blind",
    "Video-Category", "Round",
    "Answer", "Correct", "Permutation", "Position",
]


def safe_get(d: dict, key: str, default=""):
    return d.get(key, default) if isinstance(d, dict) else default


def extract_effect_and_video_category_from_part_one_image_path(image_path: str):
    parts = Path(image_path).parts
    effect = ""
    filename = ""

    if len(parts) >= 4:
        try:
            idx = parts.index("part_one")
            effect = parts[idx + 1]
            filename = parts[idx + 2]
        except ValueError:
            effect = Path(image_path).parent.name
            filename = Path(image_path).name
    else:
        effect = Path(image_path).parent.name
        filename = Path(image_path).name

    stem = Path(filename).stem
    category_only = stem.split("-")[0] if "-" in stem else stem
    return effect, category_only


def extract_effect_and_video_category_from_part_two_answer_path(answer_path: str):
    parts = Path(answer_path).parts
    effect = ""
    category = ""
    try:
        idx = parts.index("part_two")
        effect = parts[idx + 1]
        category = parts[idx + 2]
    except ValueError:
        p = Path(answer_path)
        category = p.parent.name
        effect = p.parent.parent.name
    return effect, category


def find_user_answer_files(root_dir: Path):
    for child in sorted(root_dir.iterdir()):
        if child.is_dir():
            answers_file = child / "answers.json"
            if answers_file.exists():
                yield child.name, answers_file


def main():
    parser = argparse.ArgumentParser(
        description="Convert per-user answers.json files into part_one.csv and part_two.csv."
    )
    parser.add_argument(
        "--root_dir",
        help="Root directory containing per-user subfolders with answers.json",
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Output directory for CSV files (default: current directory)",
    )
    parser.add_argument(
        "--part-one-name",
        default="part_one.csv",
        help="Filename for part_one CSV (default: part_one.csv)",
    )
    parser.add_argument(
        "--part-two-name",
        default="part_two.csv",
        help="Filename for part_two CSV (default: part_two.csv)",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    part_one_rows = []
    part_two_rows = []

    for user_id, answers_file in find_user_answer_files(root_dir):
        with open(answers_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        demo = safe_get(data, "demographics", {})
        age = safe_get(demo, "age", "")
        education = safe_get(demo, "education", "")
        frequency = safe_get(demo, "frequency", "")
        color_blind = safe_get(demo, "color_blind", "")

        # --- Part One ---
        part_one = safe_get(data, "part_one", [])
        for idx, entry in enumerate(part_one, start=1):
            image_path = safe_get(entry, "image_path", "")
            answer_text = safe_get(entry, "answer", "")

            effect, category_only = extract_effect_and_video_category_from_part_one_image_path(image_path)

            row = {
                "Effect": effect,
                "User (ID)": user_id,
                "Age": age,
                "Education": education,
                "Frequency": frequency,
                "Color Blind": color_blind,
                "Video-Category": category_only,
                "Round": idx,
                "Answer": answer_text,
                "A1": "", "A2": "", "A3": "",
                "M1": "", "M2": "", "M3": "", "M4": "",
                "L1": "", "L2": "",
                "S1": "", "S2": "",
            }
            part_one_rows.append(row)

        part_two = safe_get(data, "part_two", [])
        for idx, entry in enumerate(part_two, start=1):
            answer_path = safe_get(entry, "answer_path", "")
            effect, category = extract_effect_and_video_category_from_part_two_answer_path(answer_path)

            ans_val = safe_get(entry, "answer", "")
            corr_val = safe_get(entry, "correct", "")
            perm_val = safe_get(entry, "permutation", "")
            pos_val = safe_get(entry, "position", "")

            row = {
                "Effect": effect,
                "User (ID)": user_id,
                "Age": age,
                "Education": education,
                "Frequency": frequency,
                "Color Blind": color_blind,
                "Video-Category": category,
                "Round": idx,
                "Answer": ans_val,
                "Correct": corr_val,
                "Permutation": perm_val,
                "Position": pos_val,
            }
            part_two_rows.append(row)

    part_one_path = out_dir / args.part_one_name
    part_two_path = out_dir / args.part_two_name

    with open(part_one_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PART_ONE_COLUMNS)
        writer.writeheader()
        writer.writerows(part_one_rows)

    with open(part_two_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PART_TWO_COLUMNS)
        writer.writeheader()
        writer.writerows(part_two_rows)

    print(f"Wrote {len(part_one_rows)} rows to: {part_one_path}")
    print(f"Wrote {len(part_two_rows)} rows to: {part_two_path}")


if __name__ == "__main__":
    main()
