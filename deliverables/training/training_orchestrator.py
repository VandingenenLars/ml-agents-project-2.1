"""
data_processor.py

Description:
    This module orchestrates a compete Unity ML-Agents training run while automatically collecting system-level and training-level metrics.
Usage:
    Run this file by executing the following command:
    python training_orchestrator.py --unity-app {path_to_exe}

Author:
    Lars Vandingenen
Date:
     2025-11-10
"""
import argparse
import os
import time
import json

from pkg_resources import require

from deliverables.training.data_collector import data_collector
from deliverables.training.configs.agents.ppo.ppo_config import ppo_config

class training_orchestrator:

    # Flow:
    # 1. Create YAML config file
    # 2. run training
    # 3. use datacollector class

    def __init__(
        self,
        unity_env_path: str,
        behavior_name: str = "3DBall",
        target_reward: float = 1.0,
        system_metrics_interval: float = 2.0,
        no_graphics: bool = True
    ):
        self.unity_env_path = unity_env_path
        self.behavior_name = behavior_name
        self.target_reward = target_reward
        self.system_metrics_interval = system_metrics_interval
        self.no_graphics = no_graphics

        self.yaml_path = os.path.join(os.getcwd(), "dummy_mlagents_config.yaml")
        self.base_data_path = os.path.join(os.getcwd(), "deliverables/data/raw")

    def create_yaml(self):
        ppo_cfg = ppo_config(self.behavior_name)
        ppo_cfg.set_random_hyperparameters()
        ppo_cfg.validate_settings()
        ppo_cfg.save_config(self.yaml_path)

        print(f"[training_orchestrator] YAML created: {self.yaml_path}")
        return ppo_cfg


    def run_training(self):
        cfg = self.create_yaml()
        run_id = f"run_{int(time.time())}"

        run_path = os.path.join(self.base_data_path, run_id)
        os.makedirs(run_path, exist_ok=True)

        collector = data_collector(
            config=cfg,
            run_id=run_id,
            target_reward=self.target_reward,
            system_metrics_interval=self.system_metrics_interval,
            no_graphics=self.no_graphics,
            env_file=self.unity_env_path,
            run_path=run_path
        )


        print(f"Starting training with run ID = {run_id}")
        print(f"Output directory: {run_path}")

        results = collector.run_complete_experiment(
            config_file=self.yaml_path,
            env_file=self.unity_env_path,
            force=True
        )

        print("\n--- Training Summary ---")
        print(json.dumps(results, indent=4))

        if os.path.exists(self.yaml_path):
            os.remove(self.yaml_path)

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description= "Run Unity ML-Agents training with data collection"
    )
    parser.add_argument(
        "--unity-app",
        type=str,
        required=True,
        help="Path to the built Unity environment executable"
    )
    parser.add_argument(
        "--no-graphics",
        action="store_true",
        help="Run unity environment in no-graphics mode"
    )
    parser.add_argument(
        "--target-reward",
        type=float,
        default=1.0,
        help="Target reward threshold"
    )


    args = parser.parse_args()
    orch = training_orchestrator(
        unity_env_path=args.unity_app,
        behavior_name="3DBall",
        target_reward=1.0,
        system_metrics_interval=2.0,
        no_graphics=True
    )

    orch.run_training()
