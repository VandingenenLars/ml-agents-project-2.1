"""
yaml_config.py

Description:
    abstract class the provides the structure and shared hyperperameters for both ppo
    and sac yaml config classes

Usage:
    this module is used by both ppo_config.py and sac_config.py so you
    do not address this class directly

Author:
    Denis
Date:
    2025-09-29
"""

from abc import ABC, abstractmethod

class yaml_config(ABC):
   def __init__(self, game_type, algorithm):

       self.game_type = game_type
       self.algorithm = algorithm

       # Default Hyperperameters
       self.learning_rate = 3e-4
       self.buffer_size = 50000
       self.num_epoch = 10
       self.num_units = 64
       self.num_layers = 2
       self.max_steps = 500000

       # Default system metrics
       self.cpu_cores = 4
       self.total_ram_gb = 16.0
       self.target_mean_reward = 100
       self.gpu_utilization = 0.0
       self.cpu_utilization = 0.0

   @abstractmethod
   def save_config(self):
       """"""

   @abstractmethod
   def parse_hyperperameters(self):
       """"""

   @abstractmethod
   def validate_settings(self):
       """"""

   @abstractmethod
   def set_random_hyperparameters(self):
       """"""

   def get_yaml_config(self):
       """"""
       return self.to_yaml_dict()

   def set_manual_hyperparameters(self, params):
       """
       Manually update hyperparameters from a dictionary.
       """
       for name in params:
           # check if this attribute exists in the class
           if name in self.__dict__:
               # update its value
               self.__dict__[name] = params[name]
           else:
               print("invalid parameter name")

   def parse_network_settings(self) -> dict:
       """"""
       return {
           "normalize": True,
           "hidden_units": self.num_units,
           "num_layers": self.num_layers,
           "vis_encode_type": "simple"
       }

   def parse_reward_settings(self) -> dict:
       """"""
       return {
           "extrinsic": {
               "gamma": 0.995,
               "strength": 1.0
           }
       }
   def to_yaml_dict(self) -> dict:
       """"""
       return {
           "behaviors": {
               self.game_type: {
                   "trainer_type": self.algorithm,
                   "hyperparameters": self.parse_hyperperameters(),
                   "network_settings": self.parse_network_settings(),
                   "reward_settings": self.parse_reward_settings(),
                   "keep_checkpoints": 5,
                   "max_steps": self.max_steps,
                   "time_horizon": 1000,
                   "summary_freq": 30000
               }
           }
       }



