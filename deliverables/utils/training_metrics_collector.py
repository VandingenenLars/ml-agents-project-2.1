"""
training_metrics_collector.py

Description:
    Collects training metrics from ML-Agents TensorBoard logs.

Usage:
    Used by data_collector to monitor training progress.

Author:
    Lars
Date:
    YYYY-MM-DD
"""
import subprocess
import os
import json
import sys
import time
from pathlib import Path
import pandas as pd
from tbparse import SummaryReader

class training_metrics_collector:

    DYNAMIC_METRIC_TAGS = {
        'Cumulative Reward': 'Environment/Cumulative Reward',
        'Episode Length': 'Environment/Episode Length',
        'Policy Loss': 'Losses/Policy Loss',
        'Value Loss': 'Losses/Value Loss',
        'Learning Rate': 'Policy/Learning Rate',
        'Entropy': 'Policy/Entropy'
    }

    def __init__(self, run_id: str, run_dir: str, port: int, target_reward: float = None):
        self.port = port
        self.run_id = run_id
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # ML-Agents writes logs to results/<run_id>
        self.mlagents_results_dir = Path("results") / self.run_id
        self.target_reward = target_reward

    def run_training(self, config_file: str, env_file: str = None, force: bool = False, no_graphics: bool = True):
        command = [
            "mlagents-learn",
            f"{config_file}",
            f"--run-id={self.run_id}",
            f"--base-port={self.port}",
            "--train",
            "--torch-device=cuda"
        ]

        if env_file:
            command.append(f"--env={env_file}")
        if force:
            command.append("--force")
        if no_graphics:
            command.append("--no-graphics") 

        print(f"Starting ML-Agents training with command: {' '.join(command)}")
        try:
            process = subprocess.Popen(command)
            return process
        except FileNotFoundError:
            print(f"Error: 'mlagents-learn' not found. Install ML-Agents.", file=sys.stderr)
            raise

    def collect_scalar_metrics(self) -> pd.DataFrame:
        """
        Collects scalar metrics using tensorboard
        """
        tensorboard_dir = self.mlagents_results_dir
        
        if not tensorboard_dir.exists():
            print(f"Warning: TensorBoard directory not found: {tensorboard_dir}")
            return pd.DataFrame()

        try:
            print(f"Reading TensorBoard logs from: {tensorboard_dir}")
            reader = SummaryReader(str(tensorboard_dir), pivot=True)
            df = reader.scalars
            
            if df.empty:
                print("Warning: No scalar metrics found in TensorBoard logs")
                return pd.DataFrame()

            print(f"Available columns: {df.columns.tolist()}")
            
            rename_dict = {}
            for display_name, tag in self.DYNAMIC_METRIC_TAGS.items():
                if tag in df.columns:
                    rename_dict[tag] = display_name

            df = df.rename(columns=rename_dict)
            
            metrics_csv = self.run_dir / "training_metrics.csv"
            df.to_csv(metrics_csv, index=False)
            print(f"Saved training metrics CSV to: {metrics_csv}")
            
            metrics_json = self.run_dir / "training_metrics.json"
            
            metrics_list = df.to_dict('records')
            
            with open(metrics_json, 'w') as f:
                json.dump({
                    "run_id": self.run_id,
                    "total_steps": int(df['step'].iloc[-1]) if 'step' in df.columns else None,
                    "metrics": metrics_list
                }, f, indent=2)
            
            print(f"Saved training metrics JSON to: {metrics_json}")
            
            return df
            
        except Exception as e:
            print(f"Error reading TensorBoard logs: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def calculate_target_metrics(self, scalar_df: pd.DataFrame, target_reward: float) -> dict:
        target_metrics = {"iterations_to_target": -1, "training_time_to_target": -1.0}
        if scalar_df.empty or target_reward is None:
            return target_metrics

        if 'Cumulative Reward' not in scalar_df.columns:
            print("Warning: 'Cumulative Reward' column not found in metrics")
            return target_metrics

        reached_target = scalar_df[scalar_df['Cumulative Reward'] >= target_reward]

        if not reached_target.empty:
            first_target_row = reached_target.iloc[0]
            target_metrics["iterations_to_target"] = int(first_target_row['step']) if 'step' in first_target_row else -1
            
            if 'wall_time' in scalar_df.columns:
                start_time = scalar_df['wall_time'].min()
                elapsed_time_sec = first_target_row['wall_time'] - start_time
                target_metrics["training_time_to_target"] = round(elapsed_time_sec, 2)

        return target_metrics