"""
training_orchestrator.py

Description:
   uses the generated yaml config and Ml-agent to run the corresponding training algorithm
   ensuring everything initalized and running simultaneously

Usage:
    Example of how to use this module.

Author:
    Denis
Date:
    2025-11-06
"""

import yaml
from deliverables.training.configs.yaml_config import yaml_config
class TrainingOrchestrator:
    def __init__(self):
        self._stringID = ""
        self._yaml = yaml_config.getyaml

    def start_training(self):
        pass

    def call_ml_agent(self):
        pass

    def monitor_training(self):
        pass

    def stop_training(self):
        pass




