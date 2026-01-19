import os
import json
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from deliverables.utils.system_metrics_collector import SystemMetricsCollector
from deliverables.utils.training_metrics_collector import training_metrics_collector

class data_collector:
    def __init__(self, config, run_id=None, target_reward=None,
                 system_metrics_interval=5.0, tensorboard_port=6006,
                 env_file=None, no_graphics=True,
                 run_path=None):
        self.run_id = run_id or f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        self.config = config
        self.target_reward = target_reward
        self.system_metrics_interval = system_metrics_interval
        self.tensorboard_port = tensorboard_port
        self.env_file = env_file
        self.no_graphics = no_graphics

        self.run_dir = Path(run_path) if run_path else Path("deliverables/data/raw") / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.system_collector = None
        self.training_collector = None
        self.metrics_thread = None

    def _start_system_metrics_collection(self) -> None:
        self.system_collector = SystemMetricsCollector(
            run_dir=str(self.run_dir),
            interval=self.system_metrics_interval
        )
        self.metrics_thread = threading.Thread(target=self.system_collector.run, daemon=True)
        self.metrics_thread.start()

    def _stop_system_metrics_collection(self, grace_period: float = 5.0) -> None:
        if self.system_collector:
            self.system_collector.stop()
            if self.metrics_thread and self.metrics_thread.is_alive():
                self.metrics_thread.join(timeout=grace_period)

    def run_complete_experiment(self, config_file: str, env_file: str = None, force: bool = False) -> Dict[str, Any]:
        self._start_system_metrics_collection()

        self.training_collector = training_metrics_collector(
            run_id=self.run_id,
            run_dir=str(self.run_dir),
            port=self.tensorboard_port,
            target_reward=self.target_reward
        )
        
        ml_agents_process = None
        try:
            ml_agents_process = self.training_collector.run_training(
                config_file=config_file,
                env_file=env_file or self.env_file,
                force=force,
                no_graphics=self.no_graphics
            )
            ml_agents_process.wait()

            if ml_agents_process.returncode == 0:
                final_metrics = self.training_collector.collect_scalar_metrics()
                target_metrics = self.training_collector.calculate_target_metrics(final_metrics, self.target_reward)
                
                final_reward = None
                if not final_metrics.empty and 'mean_reward' in final_metrics.columns:
                    final_reward = float(final_metrics['mean_reward'].iloc[-1])
                
                summary = {
                    "run_id": self.run_id,
                    "training_completed_successfully": True,
                    "final_cumulative_reward": final_reward,
                    "target_metrics": target_metrics,
                    "total_steps": int(final_metrics['steps'].iloc[-1]) if not final_metrics.empty else None
                }
                
                with open(self.run_dir / "summary.json", 'w') as f:
                    json.dump(summary, f, indent=4)
                return summary
            else:
                return {"run_id": self.run_id, "training_completed_successfully": False, "error_code": ml_agents_process.returncode}

        except KeyboardInterrupt:
            if ml_agents_process:
                ml_agents_process.terminate()
            raise
        finally:
            self._stop_system_metrics_collection()