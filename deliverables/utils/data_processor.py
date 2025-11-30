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

def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    run_folder = Path("deliverables/data/raw/mock_run_2025-11-10_12-00-00")
    processed_folder = Path("deliverables/data/processed")
    processed_folder.mkdir(parents=True, exist_ok=True)

    # Load config.json
    config = load_json(run_folder / "config.json")
    target_mean_reward = config.get("target_mean_reward", 100)

    # Load metrics
    system_metrics = load_json(run_folder / "system_metrics.json")
    training_metrics = load_json(run_folder / "training_metrics.json")

    # Compute iterations_to_target and training_time_to_target_sec
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

    # Prepare CSV
    run_id = run_folder.name.replace("mock_", "run_")
    num_rows = min(len(system_metrics), len(training_metrics))
    output_file = processed_folder / f"{run_id}.csv"

    fieldnames = [
        "run_id","game_type","algorithm","learning_rate","batch_size","buffer_size",
        "num_epoch","num_units","num_layers","max_steps","cpu_cores","total_ram_gb",
        "avg_cpu_utilization","avg_gpu_utilization","avg_ram_usage_mb","peak_ram_usage_mb",
        "target_mean_reward","iterations_to_target","training_time_to_target_sec"
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(num_rows):
            sys_row = system_metrics[i]
            row = {
                "run_id": run_id,
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
                "avg_cpu_utilization": sys_row.get("cpu_utilization"),
                "avg_gpu_utilization": sys_row.get("gpu_utilization"),
                "avg_ram_usage_mb": sys_row.get("ram_used_mb"),
                "peak_ram_usage_mb": sys_row.get("ram_used_mb"),
                "target_mean_reward": target_mean_reward,
                "iterations_to_target": iterations_to_target,
                "training_time_to_target_sec": training_time_to_target_sec
            }
            writer.writerow(row)

    print(f"CSV created at {output_file}")

if __name__ == "__main__":
    main()
