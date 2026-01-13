"""
data_processor.py

Description:
    Describe what this module does.

Usage:
    Example of how to use this module.

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
        return None

    target_mean_reward = config.get("target_mean_reward", 100)

    avg_cpu = mean(m["cpu_utilization"] for m in system_metrics)
    avg_gpu = mean(m["gpu_utilization"] for m in system_metrics)
    avg_ram = mean(m["ram_used_mb"] for m in system_metrics)
    peak_ram = max(m["ram_used_mb"] for m in system_metrics)

    iterations_to_target = 0
    training_time_to_target_sec = 0.0
    first_time = datetime.fromisoformat(training_metrics[0]["timestamp"])

    for tm in training_metrics:
        mean_reward = tm.get("mean_reward")
        if mean_reward is not None and mean_reward >= target_mean_reward:
            iterations_to_target = tm.get("steps", 0)
            current_time = datetime.fromisoformat(tm["timestamp"])
            training_time_to_target_sec = (current_time - first_time).total_seconds()
            break

    return {
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
        "training_time_to_target_sec": training_time_to_target_sec
    }

def main():
    raw_folder = Path("deliverables/data/raw")
    processed_folder = Path("deliverables/data/processed")
    processed_folder.mkdir(parents=True, exist_ok=True)
    output_file = processed_folder / "processed_data.csv"

    run_folders = [
        p for p in raw_folder.iterdir()
        if p.is_dir() and p.name.startswith("run")
    ]

    valid_runs = []

    for folder in run_folders:
        result = process_run(folder)
        if result is not None:
            valid_runs.append(result)
    if not valid_runs:
        return

    aggregated = {
        "game_type": valid_runs[0]["game_type"],
        "algorithm": valid_runs[0]["algorithm"],
        "learning_rate": mean(r["learning_rate"] for r in valid_runs),
        "batch_size": mean(r["batch_size"] for r in valid_runs),
        "buffer_size": mean(r["buffer_size"] for r in valid_runs),
        "num_epoch": mean(r["num_epoch"] for r in valid_runs),
        "num_units": mean(r["num_units"] for r in valid_runs),
        "num_layers": mean(r["num_layers"] for r in valid_runs),
        "max_steps": mean(r["max_steps"] for r in valid_runs),
        "cpu_cores": mean(r["cpu_cores"] for r in valid_runs),
        "total_ram_gb": mean(r["total_ram_gb"] for r in valid_runs),
        "avg_cpu_utilization": mean(r["avg_cpu_utilization"] for r in valid_runs),
        "avg_gpu_utilization": mean(r["avg_gpu_utilization"] for r in valid_runs),
        "avg_ram_usage_mb": mean(r["avg_ram_usage_mb"] for r in valid_runs),
        "peak_ram_usage_mb": max(r["peak_ram_usage_mb"] for r in valid_runs),
        "target_mean_reward": mean(r["target_mean_reward"] for r in valid_runs),
        "iterations_to_target": mean(r["iterations_to_target"] for r in valid_runs),
        "training_time_to_target_sec": mean(r["training_time_to_target_sec"] for r in valid_runs)
    }

    fieldnames = list(aggregated.keys())

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(aggregated)

if __name__ == "__main__":
    main()
