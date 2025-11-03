from deliverables.training.configs.yaml_config import yaml_config
import os
import yaml
class sac_config(yaml_config):
    def __init__(self,game_type):
        super().__init__(game_type,"sac")
        self.batch_size = 256
        self.buffer_init_steps = 0
        self.tau = 0.005
        self.steps_per_update = 20.0
        self.save_replay_buffer = False
        self.int_entcoef = 1.0
        self.reward_signal_steps_per_update = 20.0
        self.learning_rate_schedule = "constant"


    def load_config(self):
        """"""
        directory_path = os.path.dirname(__file__)
        file_path = os.path.join(directory_path,"mock_config.yaml")

        with open(file_path,"w") as file:
            yaml.dump(self.to_yaml_dict(),file,sort_keys=False)

        print("updated SAC mock_config.yaml")

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
            "buffer_init_steps": self.buffer_init_steps,
            "tau": self.tau,
            "steps_per_update": self.steps_per_update,
            "save_replay_buffer": self.save_replay_buffer,
            "int_entcoef": self.int_entcoef,
            "reward_signal_steps_per_update": self.reward_signal_steps_per_update,
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
        if self.tau <=0:
            raise ValueError("tau must be positiv")


if __name__ == "__main__":

    sacConfig = sac_config("walker")
    sacConfig.validate_settings()
    sacConfig.load_config()
