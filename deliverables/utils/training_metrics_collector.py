import subprocess
import os
import json
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
from tbparse import SummaryReader

class training_metrics_collector:
    """
    Collects training metrics from ML-Agents TensorBoard logs and syncs timestamps.
    """
    DYNAMIC_METRIC_TAGS = {
        'mean_reward': 'Environment/Cumulative Reward',
        'loss': 'Losses/Policy Loss',
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
            "--torch-device=cpu"
        ]
        if env_file: command.append(f"--env={env_file}")
        if force: command.append("--force")
        if no_graphics: command.append("--no-graphics")

        print(f"Starting ML-Agents training with command: {' '.join(command)}")
        try:
            return subprocess.Popen(command)
        except FileNotFoundError:
            print(f"Error: 'mlagents-learn' not found.", file=sys.stderr)
            raise

    def collect_scalar_metrics(self) -> pd.DataFrame:
        if not self.mlagents_results_dir.exists():
            print(f"Warning: TensorBoard directory not found: {self.mlagents_results_dir}")
            return pd.DataFrame()

        try:
            print(f"Reading TensorBoard logs from: {self.mlagents_results_dir}")
            
            reader = SummaryReader(str(self.mlagents_results_dir))
            df_raw = reader.scalars

            if df_raw.empty:
                return pd.DataFrame()

            
            df_pivot = df_raw.pivot(index='step', columns='tag', values='value').reset_index()
            
            
            rename_dict = {
                'step': 'steps',
                self.DYNAMIC_METRIC_TAGS['mean_reward']: 'mean_reward',
                self.DYNAMIC_METRIC_TAGS['loss']: 'loss'
            }
            df_pivot = df_pivot.rename(columns={k: v for k, v in rename_dict.items() if k in df_pivot.columns})

            # calculate time using elapsed time
            start_time_epoch = os.path.getctime(self.mlagents_results_dir)
            
            
            current_time_epoch = datetime.now().timestamp()
            total_duration = current_time_epoch - start_time_epoch
            max_steps = df_pivot['steps'].max() if not df_pivot.empty else 1

            def calculate_timestamp(step):
                elapsed = (step / max_steps) * total_duration
                actual_time = start_time_epoch + elapsed
                return datetime.fromtimestamp(actual_time).strftime('%Y-%m-%dT%H:%M:%S')

            df_pivot['timestamp'] = df_pivot['steps'].apply(calculate_timestamp)

            target_cols = ['timestamp', 'mean_reward', 'steps', 'loss']
            df_final = df_pivot[[c for c in target_cols if c in df_pivot.columns]].copy()

            metrics_json = self.run_dir / "training_metrics.json"
            with open(metrics_json, 'w') as f:
                json.dump(df_final.to_dict('records'), f, indent=2)

            print(f"Saved synchronized training metrics with calculated timestamps: {metrics_json}")
            return df_final

        except Exception as e:
            print(f"Error processing metrics: {e}")
            return pd.DataFrame()

    def calculate_target_metrics(self, scalar_df: pd.DataFrame, target_reward: float) -> dict:
        target_metrics = {"iterations_to_target": -1}
        if scalar_df.empty or target_reward is None or 'mean_reward' not in scalar_df.columns:
            return target_metrics

        reached = scalar_df[scalar_df['mean_reward'] >= target_reward]
        if not reached.empty:
            target_metrics["iterations_to_target"] = int(reached.iloc[0]['steps'])
        
        return target_metrics