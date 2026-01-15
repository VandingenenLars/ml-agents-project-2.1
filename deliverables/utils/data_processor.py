"""
data_processor.py

Description:
    This code takes and loads 3 json files from a folder, each folder being a new run and procceses it. Than the data is worked into one single csv line.
    This is done with multiple folder runs so that at the end there is 1 csv file with the data from all the runs, being ready to be worked on.

Usage:
    In order for this code to work there is needed to be at least one folder run.

Author:
    Jianu Mihnea-Alexandru
Date:
     2025-11-10
"""
import json
import csv
from pathlib import Path
from datetime import datetime
from statistics import mean

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

    if not config or not system_metrics or not training_metrics:
        print(f"Skipping {run_folder.name}: missing files")
        return None

    target_mean_reward = config.get("target_mean_reward", 100)

    avg_cpu = mean(m["cpu_utilization"] for m in system_metrics)
    avg_gpu = mean(m["gpu_utilization"] for m in system_metrics)
    avg_ram = mean(m["ram_used_mb"] for m in system_metrics)
    peak_ram = max(m["ram_used_mb"] for m in system_metrics)

    iterations_to_target = None
    for tm in training_metrics:
        mean_reward = tm.get("mean_reward")
        if mean_reward is not None and mean_reward >= target_mean_reward:
            iterations_to_target = tm.get("step")
            break

    row = {
        "run_id" : run_folder.name,
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
        "avg_gpu_utilization": avg_gpu,
        "avg_ram_usage_mb": avg_ram,
        "peak_ram_usage_mb": peak_ram,
        "target_mean_reward": target_mean_reward,
        "iterations_to_target": iterations_to_target,
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
