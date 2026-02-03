import json
import os
from pathlib import Path

folder = "results"
results_folders = os.listdir(folder)

video_total = {}
effect_total = {}


for results_folder in results_folders:
    answer_json = os.path.join(folder, results_folder, "answers.json")
    with open(answer_json, "r") as f:
        answer = json.load(f)
        part_one = answer["part_one"]

        for i in range(5):
            part_two_slice = part_one[i]
            image_path = Path(part_two_slice["image_path"])

            video = image_path.stem
            effect = image_path.parent.stem
            answer = part_two_slice["answer"]

            if video in video_total:
                video_total[video].append(answer)
            else:
                video_total[video] = [answer]
            if video in effect_total:
                if effect in effect_total[video]:
                    effect_total[video][effect].append(answer)
                else:
                    effect_total[video][effect] = [answer]
            else:
                effect_total[video] = {
                    effect: [answer]
                }

results = {
    "video_total": video_total,
    "effect_total": effect_total
}

with open("part_one_answers.json", "w") as f:
    json.dump(results, f)
