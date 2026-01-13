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
import yaml
import csv
from pathlib import Path
from datetime import datetime
from statistics import mean
from typing import Any, Dict, List, Optional


def load_json(path: Path) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return None
            if content.startswith("{") and "\n{" in content:
                return [json.loads(line) for line in content.splitlines() if line.strip()]
            return json.loads(content)
    except Exception:
        return None


def load_yaml(path: Path) -> Optional[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def safe_mean(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return mean(vals) if vals else None


def safe_max(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return max(vals) if vals else None


def parse_iso(s: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(s.rstrip("Z"))
    except Exception:
        return None


def process_run(run_folder: Path, yaml_root: Path) -> Optional[Dict]:
    system_path = run_folder / "system_metrics.json"
    training_path = run_folder / "training_metrics.json"
    summary_path = run_folder / "summary.json"
    yaml_path = yaml_root / run_folder.name / "configuration.yaml"

    system_metrics = load_json(system_path)
    training_data = load_json(training_path)
    summary = load_json(summary_path)
    config = load_yaml(yaml_path)

    if system_metrics is None or training_data is None or config is None:
        return None

    if isinstance(training_data, dict) and "metrics" in training_data:
        training_metrics = training_data["metrics"]
    elif isinstance(training_data, list):
        training_metrics = training_data
    else:
        return None

    if not system_metrics or not training_metrics:
        return None

    cpu_avg = mean(m.get("cpu_utilization", 0) for m in system_metrics)
    ram_used_avg = mean(m.get("ram_used_mb", 0) for m in system_metrics)
    gpu_avg = mean(m.get("gpu_utilization", 0) for m in system_metrics)
    peak_ram = max(m.get("ram_used_mb", 0) for m in system_metrics)

    target_mean_reward = config.get("target_mean_reward", 100)
    iterations_to_target = None
    training_time_to_target_sec = None

    if summary and "target_metrics" in summary:
        iterations_to_target = summary["target_metrics"].get("iterations_to_target")
        training_time_to_target_sec = summary["target_metrics"].get("training_time_to_target")

    if iterations_to_target is None or training_time_to_target_sec is None:
        first_time = None
        for entry in training_metrics:
            for k in ("timestamp", "time", "wall_time", "datetime"):
                if k in entry and isinstance(entry[k], str):
                    first_time = parse_iso(entry[k])
                    if first_time:
                        break
            if first_time:
                break

        if first_time:
            for tm in training_metrics:
                reward = tm.get("mean_reward") or tm.get("reward") or tm.get("Cumulative Reward")
                if reward is not None and reward >= target_mean_reward:
                    iterations_to_target = tm.get("steps") or tm.get("iteration") or tm.get("step")
                    for k in ("timestamp", "time", "wall_time", "datetime"):
                        if k in tm and isinstance(tm[k], str):
                            ts = parse_iso(tm[k])
                            if ts:
                                training_time_to_target_sec = (ts - first_time).total_seconds()
                                break
                    break

    behavior = config.get("behaviors", {}).get("3DBall", {})
    hyper = behavior.get("hyperparameters", {})
    network = behavior.get("network_settings", {})

    return {
        "game_type": behavior.get("trainer_type"),
        "algorithm": behavior.get("trainer_type"),
        "learning_rate": hyper.get("learning_rate"),
        "batch_size": hyper.get("batch_size"),
        "buffer_size": hyper.get("buffer_size"),
        "num_epoch": hyper.get("num_epoch"),
        "num_units": network.get("hidden_units"),
        "num_layers": network.get("num_layers"),
        "max_steps": behavior.get("max_steps"),
        "cpu_cores": config.get("cpu_cores"),
        "total_ram_gb": config.get("total_ram_gb"),
        "avg_cpu_utilization": cpu_avg,
        "avg_gpu_utilization": gpu_avg,
        "avg_ram_usage_mb": ram_used_avg,
        "peak_ram_usage_mb": peak_ram,
        "target_mean_reward": target_mean_reward,
        "iterations_to_target": iterations_to_target,
        "training_time_to_target_sec": training_time_to_target_sec,
    }


def main():
    project_root = Path(__file__).resolve().parents[2]
    raw_root = project_root / "deliverables" / "data" / "raw"
    results_root = project_root / "results"

    runs = []
    for run in raw_root.iterdir():
        if run.is_dir():
            r = process_run(run, results_root)
            if r:
                runs.append(r)

    if not runs:
        return

    def g(k):
        return [r[k] for r in runs if r.get(k) is not None]

    aggregated = {
        "game_type": runs[0]["game_type"],
        "algorithm": runs[0]["algorithm"],
        "learning_rate": safe_mean(g("learning_rate")),
        "batch_size": safe_mean(g("batch_size")),
        "buffer_size": safe_mean(g("buffer_size")),
        "num_epoch": safe_mean(g("num_epoch")),
        "num_units": safe_mean(g("num_units")),
        "num_layers": safe_mean(g("num_layers")),
        "max_steps": safe_mean(g("max_steps")),
        "cpu_cores": safe_mean(g("cpu_cores")),
        "total_ram_gb": safe_mean(g("total_ram_gb")),
        "avg_cpu_utilization": safe_mean(g("avg_cpu_utilization")),
        "avg_gpu_utilization": safe_mean(g("avg_gpu_utilization")),
        "avg_ram_usage_mb": safe_mean(g("avg_ram_usage_mb")),
        "peak_ram_usage_mb": safe_max(g("peak_ram_usage_mb")),
        "target_mean_reward": safe_mean(g("target_mean_reward")),
        "iterations_to_target": safe_mean(g("iterations_to_target")),
        "training_time_to_target_sec": safe_mean(g("training_time_to_target_sec")),
    }

    out_dir = project_root / "deliverables" / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "processed_data.csv"

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=aggregated.keys())
        writer.writeheader()
        writer.writerow({k: v if v is not None else "" for k, v in aggregated.items()})


if __name__ == "__main__":
    main()
