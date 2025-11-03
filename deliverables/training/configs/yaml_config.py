from abc import ABC, abstractmethod

class yaml_config(ABC):
   def __init__(self, game_type, algorithm):
       self.game_type = game_type
       self.algorithm = algorithm

       # hyperperameters
       self.learning_rate = 3e-4
       self.buffer_size = 50000
       self.num_epoch = 10
       self.num_units = 64
       self.num_layers = 2
       self.max_steps = 500000

       # system metrics
       self.cpu_cores = 4
       self.total_ram_gb = 16.0
       self.target_mean_reward = 100
       self.gpu_utilization = 0.0
       self.cpu_utilization = 0.0

   @abstractmethod
   def load_config(self):
       """"""
       pass

   @abstractmethod
   def parse_hyperperameters(self):
       """"""
       pass
   @abstractmethod
   def validate_settings(self):
       """"""
       pass

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



