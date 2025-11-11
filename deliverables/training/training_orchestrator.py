"""
training_orchestrator.py

Description:
   uses the generated yaml config and Ml-agent to run the corresponding training algorithm
   ensuring everything initialised and running simultaneously

Usage:
    Example of how to use this module.

Author:
    Denis
Date:
    2025-11-06
"""

import yaml
from deliverables.training.configs.yaml_config import yaml_config
from deliverables.training.configs.agents.ppo.ppo_config import ppo_config
from deliverables.training.configs.agents.sac.sac_config import sac_config

class training_orchestrator:
    def __init__(self,config):
        self.run_id = ""
        self.config = config
        self.yaml_path = ""

    def get_yaml_dict(self):
        """"""
        return self.config.get_yaml_config()

    def start_training(self):
        """"""
        pass

    def call_ml_agent(self):
        """"""
        pass

    def monitor_training(self):
        """"""
        pass

    def stop_training(self):
        """"""
        pass


if __name__ == "__main__":
    game_type = input(print("Enter game type: "))
    alg_type = input(print("Enter algorithm type: "))
    perameter_mode = input(print("Random hyperparameter mode?: [y/n]"))

    if alg_type == "ppo":
        config = ppo_config(game_type)
    elif alg_type == "sac":
        config = sac_config(game_type)
    else:
        print("Invalid algorithm type")

    if perameter_mode == "y":
        config.set_random_hyperparameters()

    elif perameter_mode == "n":
        config.set_manual_hyperparameters()
    else:
        pass

    config.validate_settings()
    orchestator = training_orchestrator(config)




