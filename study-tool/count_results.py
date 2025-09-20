import json
import os
from pathlib import Path

folder = "results"
results_folders = os.listdir(folder)

effects_amount = {}
video_types_amount = {}
videos_amount = {}
choices_amount = {}
video_effect_combinations = {}
video_choice_combinations = {}
effect_choice_combinations = {}

for results_folder in results_folders:
    answer_json = os.path.join(folder, results_folder, "answers.json")
    with open(answer_json, "r") as f:
        answer = json.load(f)
        part_two = answer["part_two"]

        for i in range(5):
            part_two_slice = part_two[i]
            image_path = Path(part_two_slice["correct_path"])

            video = image_path.stem
            video_type = image_path.parent.stem
            effect = image_path.parent.parent.stem
            choice = part_two_slice["correct"]

            if effect in effects_amount:
                effects_amount[effect] = effects_amount[effect] + 1
            else:
                effects_amount[effect] = 1
            if video_type in video_types_amount:
                video_types_amount[video_type] = video_types_amount[video_type] + 1
            else:
                video_types_amount[video_type] = 1
            if video in videos_amount:
                videos_amount[video] = videos_amount[video] + 1
            else:
                videos_amount[video] = 1
            if choice in choices_amount:
                choices_amount[choice] = choices_amount[choice] + 1
            else:
                choices_amount[choice] = 1

            video_effect_combination = video + "_" + effect
            if video_effect_combination in video_effect_combinations:
                video_effect_combinations[video_effect_combination] = video_effect_combinations[video_effect_combination] + 1
            else:
                video_effect_combinations[video_effect_combination] = 1
            video_choice_combination = video + "_" + str(choice)
            if video_choice_combination in video_choice_combinations:
                video_choice_combinations[video_choice_combination] = video_choice_combinations[video_choice_combination] + 1
            else:
                video_choice_combinations[video_choice_combination] = 1
            effect_choice_combination = effect + "_" + str(choice)
            if effect_choice_combination in effect_choice_combinations:
                effect_choice_combinations[effect_choice_combination] = effect_choice_combinations[effect_choice_combination] + 1
            else:
                effect_choice_combinations[effect_choice_combination] = 1
        results = {
            "effects": effects_amount,
            "video_types": video_types_amount,
            "videos": videos_amount,
            "choices": choices_amount,
        }
        print(results)

results = {
    "effects": effects_amount,
    "video_types": video_types_amount,
    "videos": videos_amount,
    "choices": choices_amount,
    "video_effect_combinations": video_effect_combinations,
    "video_choice_combinations": video_choice_combinations,
    "effect_choice_combinations": effect_choice_combinations,
}

with open("results.json", "w") as f:
    json.dump(results, f)
