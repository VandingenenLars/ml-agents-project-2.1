"""
data_processor.py

Description:
    data_processor.py is responsible for processing all runs under 'data/raw/...'
    all folders structure config.json. system_metrics.json and training_metrics.json
    runs that are missing essential information are cleaned from the dataset
    variables such as training_succes and iterations_to_convergence are derived from given data
    results are saved to a single processed_data.csv under "data/processed/..." ready for feature extraction
"
Usage:
    run this module as a main class with `python data_processor.py`

Author:
    Jianu Mihnea-Alexandru / Andreas Constantinou
Date:
     2025-11-10
"""
import json
import csv
from pathlib import Path
from statistics import mean

import numpy as np

TARGET_MEAN_REWARD = 85
EARLY_FRACTION = 0.3

def load_json(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def process_run(run_folder: Path):
    config = load_json(run_folder / "config.json")
    system_metrics = load_json(run_folder / "system_metrics.json")
    training_metrics = load_json(run_folder / "training_metrics.json")

    if not config or not training_metrics:
        print(f"Skipping {run_folder.name}: missing files")
        return None

    # optional but supported / will help when using other's datasets
    if system_metrics:
        avg_cpu = mean(m["cpu_utilization"] for m in system_metrics)
        avg_ram = mean(m["ram_used_mb"] for m in system_metrics)
        peak_ram = max(m["ram_used_mb"] for m in system_metrics)
    else:
        avg_cpu = avg_ram = peak_ram = None

    cutoff = max(1, int(len(training_metrics) * EARLY_FRACTION))
    valid_entries = [tm for tm in training_metrics[:cutoff] if tm.get("cumulative_reward") is not None
                     and tm.get("step") is not None
                     and tm.get("timestamp") is not None]
    rewards = [tm["cumulative_reward"] for tm in valid_entries]
    steps = [tm["step"] for tm in valid_entries]
    timestamps = [tm["timestamp"] for tm in valid_entries]

    reached_threshold = False
    iterations_to_threshold = None
    seconds_to_threshold = None

    if len(rewards) < 2:
        # just use mean of all available rewards
        curr_avg = np.mean(rewards)
        if curr_avg >= TARGET_MEAN_REWARD:
            reached_threshold = True
            iterations_to_threshold = steps[-1]
            seconds_to_threshold = timestamps[-1] - timestamps[0]
    else:
        window = max(1, len(rewards) // 2)
        for i in range(window, len(rewards)):
            curr_avg = np.mean(rewards[i - window:i])
            if curr_avg >= TARGET_MEAN_REWARD:
                reached_threshold = True
                iterations_to_threshold = steps[i]
                seconds_to_threshold = timestamps[i] - timestamps[0]
                break

    row = {
        "run_id": run_folder.name,
        "game_type": config.get("game_type"),
        "algorithm": config.get("algorithm"),
        "learning_rate": config.get("learning_rate"),
        "batch_size": config.get("batch_size"),
        "buffer_size": config.get("buffer_size"),
        "num_epoch": config.get("num_epoch"),
        "num_units": config.get("num_units"),
        "num_layers": config.get("num_layers"),
        "max_steps": config.get("max_steps"),
        "cpu_cores": config.get("cpu_cores"),
        "total_ram_gb": config.get("total_ram_gb"),
        "avg_cpu_utilization": avg_cpu,
        "avg_ram_usage_mb": avg_ram,
        "peak_ram_usage_mb": peak_ram,
        "reached_threshold": int(reached_threshold),
        "iterations_to_threshold": iterations_to_threshold,
        "seconds_to_threshold": seconds_to_threshold
    }

    return row

def main():
    raw_folder = Path("../data/raw")
    processed_folder = Path("../data/processed")
    processed_folder.mkdir(parents=True, exist_ok=True)
    output_file = processed_folder / "processed_data.csv"

    run_folders = [
        p for p in raw_folder.iterdir()
        if p.is_dir() and p.name.startswith("run")
    ]

    rows = []

    for folder in run_folders:
        row = process_run(folder)
        if row:
            rows.append(row)

    if not rows:
        print("No valid runs found.")


    fieldnames = list(rows[0].keys())

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("Done processing all runs.")

if __name__ == "__main__":
    main()
