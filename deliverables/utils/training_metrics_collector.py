"""
training_metrics_collector.py

Description:
    Describe what this module does.

Usage:
    Example of how to use this module.

Author:
    Lars
Date:
    YYYY-MM-DD
"""

import subprocess
import os
import json
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

    def __init__(self, run_id: str, results_dir: str, port: int, target_reward: float = None):

        self.port = port
        self.run_id = run_id
        self.results_dir = Path(results_dir)
        self.log_dir = self.results_dir / self.run_id

        self.tboard_log_path = self.log_dir
        self.target_reward = target_reward

    def run_training (self, config_file:str, env_file: str = None, force: bool = False):
        command  = [
            "mlagents-learn",
            str(config_file),
            "--run-id", self.run_id,
        ]

        if env_file:
            command.extend(["--env",str(env_file)])

        if force:
            command.append("--force")
        
        try:
            subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            raise

    def collect_metrics(self) -> pd.DataFrame:
        if not self.tboard_log_path.exists():
            print("Tensorboard directory not found")
            return pd.DataFrame()
        
        reader = SummaryReader(str(self.tboard_log_path))
        df = reader.scalars
        
        # Filter for the relevant metrics
        metric_tags = list(self.DYNAMIC_METRIC_TAGS.values())
        filtered_df = df[df['tag'].isin(metric_tags)]
        
        # Pivot for easier analysis: step as index, metric tags as columns
        pivot_df = filtered_df.pivot(index='step', columns='tag', values='value').reset_index()
        pivot_df.rename(columns={v: k for k, v in self.DYNAMIC_METRIC_TAGS.items()}, inplace=True)

        # Merge in wall_time (needed for 'training time to target')
        # Wall time is recorded in seconds since the Unix epoch
        wall_time_df = df[['step', 'wall_time']].drop_duplicates(subset=['step']).set_index('step')
        pivot_df = pivot_df.set_index('step').join(wall_time_df).reset_index()

        return pivot_df
    
    def calculate_target_metrics(self, scalar_df: pd.DataFrame, target_reward: float) -> dict:
        """
        Calculates iterations to target and training time to target.
        """
        target_metrics = {
            "iterations to target": -1,
            "training time to target": -1.0 
        }
        
        if scalar_df.empty or target_reward is None:
            return target_metrics

        reward_tag = self.DYNAMIC_METRIC_TAGS['Cumulative Reward']
        
        reached_target = scalar_df[scalar_df['Cumulative Reward'] >= target_reward]
        
        if not reached_target.empty:
            first_target_row = reached_target.iloc[0]
            
            target_metrics["iterations to target"] = int(first_target_row['step'])
            
            
            start_time = scalar_df['wall_time'].min()
            
            elapsed_time_sec = first_target_row['wall_time'] - start_time
            
            target_metrics["training time to target"] = round(elapsed_time_sec, 2)
        
        return target_metrics
    
    def run_and_collect(self, config_file: str, env_file: str = None, force: bool = False):
        """Orchestrates the training run and subsequent data collection."""
        
        self.run_training(config_file, env_file, force)
        
        scalar_metrics = self.collect_scalar_metrics()
        
        target_metrics = self.calculate_target_metrics(scalar_metrics, self.target_reward)
        
        return {
            "scalars": scalar_metrics,
            "targets": target_metrics
        }