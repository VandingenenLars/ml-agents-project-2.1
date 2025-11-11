"""
training_orchestrator.py

Description:
   uses the generated yaml config and Ml-agent to run the corresponding training algorithm
<<<<<<< HEAD
   ensuring everything initialised and running simultaneously
=======
   ensuring everything initialized and running simultaneously
>>>>>>> origin/develop

Usage:
    Example of how to use this module.

Author:
    Denis
Date:
    2025-11-06
"""

import time
import subprocess
from deliverables.training.configs.agents.ppo.ppo_config import ppo_config
from deliverables.training.configs.agents.sac.sac_config import sac_config

class training_orchestrator:
    def __init__(self,config,run_id=None):

        self.run_id = run_id or self.algorithm + "_" + self.game_type + "_" + str(int(time.time()))
        self.config = config
        self.yaml_path = ""

    def get_yaml_dict(self):
        """"""
        return self.config.get_yaml_config()

    def start_training(self, env_file=None, force=False):
        """"""
        self.config.save_config()
        self.yaml_path = "mock_config.yaml"  # or wherever your save_config puts it

        # call ML-Agents to run training
        self.call_ml_agent(self.yaml_path, env_file, force)


    def call_ml_agent(self,config_file, env_file=None, force=False):
        """"""
        command = [
            "mlagents-learn",
            str(config_file),
            "--run-id", self.run_id
        ]

        if env_file:
            command.extend(["--env", str(env_file)])
        if force:
            command.append("--force")


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
        print("Random hyperparameter entry mode selected.")
        config.set_random_hyperparameters()

    elif perameter_mode == "n":

        print("Manual hyperparameter entry mode selected.")
        manual_params = {}

        while True:

            key = input("Enter a hyperparameter name (or type 'done' to finish): ")
            if key.lower() == "done":
                break

            # get the value for that hyperparameter
            value = input(f"Enter a value for {key}: ")
            manual_params[key] = value

        # apply all entered values to the config
        config.set_manual_hyperparameters(manual_params)

    config.validate_settings()
    orchestator = training_orchestrator(config)
