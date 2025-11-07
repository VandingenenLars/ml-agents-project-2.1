"""
ppo_config.py

Description:
    class that provides a yaml config for training with the ppo training algorithm,
    outputting to mock_config

Usage:
    initialize with ppoConfig = ppo_config("my_ppo_agent")
    validate with ppoConfig.validate_settings()
    load yaml config into mock_config.yaml with ppoConfig.load_config()

Author:
    Denis
Date:
    2025-09-31
"""

from deliverables.training.configs.yaml_config import yaml_config
import os
import yaml
class ppo_config(yaml_config):
    def __init__(self,game_type):
        super().__init__(game_type,"ppo")
        self.batch_size = 2048
        self.beta = 0.005
        self.epsilon = 0.02
        self.lambd = 0.95
        self.learning_rate_schedule = "linear"

    def load_config(self):
        """"""
        directory_path = os.path.dirname(__file__)
        file_path = os.path.join(directory_path,"mock_config.yaml")

        with open(file_path,"w") as file:
            yaml.dump(self.to_yaml_dict(),file,sort_keys=False)

        print("updated PPO mock_config.yaml")

    def parse_hyperperameters(self):
        """"""
        return{
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "num_epoch": self.num_epoch,
            "num_units": self.num_units,
            "num_layers": self.num_layers,
            "max_steps": self.max_steps,
            "beta": self.beta,
            "epsilon": self.epsilon,
            "lambd": self.lambd,
            "learning_rate_schedule": self.learning_rate_schedule,
            "cpu_cores": self.cpu_cores,
            "total_ram_gb": self.total_ram_gb,
            "target_mean_reward": self.target_mean_reward,
            "gpu_utilization": self.gpu_utilization,
            "cpu_utilization": self.cpu_utilization
        }
    def validate_settings(self):
        """"""
        if self.num_units <= 0:
            raise ValueError("Number of units must be positive")
        if self.num_layers <= 0:
            raise ValueError("Number of layers must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.max_steps <= 0:
            raise ValueError("Max steps must be positive")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.beta <= 0:
            raise ValueError("beta must be positive")
        if self.lambd <= 0:
            raise ValueError("lambd must be positive")
        if self.epsilon <=0:
            raise ValueError("epsilon must be positive")

if __name__ == "__main__":

    ppoConfig = ppo_config("walker")
    ppoConfig.validate_settings()
    ppoConfig.load_config()
