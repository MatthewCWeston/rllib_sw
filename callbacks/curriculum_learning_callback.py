from functools import partial
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
)

def _remote_fn(env_runner, config: dict):
    # We recreate the entire env object by changing the env_config on the worker,
    # then calling its `make_env()` method.
    env_runner.config.environment(env_config=config)
    env_runner.make_env()

class CurriculumLearningCallback(RLlibCallback):
    """
        Custom callback handling curriculum learning for our SPACEWAR! single-player environment,
        with the intention of bootstrapping a baseline for PVP.
        
         - Escalate from zero gravity to 1.0 gravity each time a threshold is met
         - After this, escalate to orbiting targets
         - Finally, escalate to targets that rotate towards the player while constantly shooting
    """
    def __init__(
        self,
        env_config: dict,
        promotion_threshold: float = 1.0,
        promotion_patience: int = 2,
        num_increments: int = 10,
        start_increment: int = 0,
    ):
        self.env_config = env_config.copy() # Adjust gravity here
        self.promotion_threshold = promotion_threshold
        self.promotion_patience = promotion_patience
        self.num_increments = num_increments
        self.start_increment = start_increment
        
    def on_algorithm_init(
        self,
        *,
        algorithm: Algorithm,
        **kwargs,
    ) -> None:
        # Set the initial task to 0.
        algorithm.metrics.log_value("current_env_task", self.start_increment, reduce="sum", window=1)
        algorithm.metrics.log_value("promotion_cycles", 0, reduce="sum", window=1)
        
    def promote(self, algorithm, current_task, metrics_logger):
        next_task = current_task + 1.0
        self.env_config['grav_multiplier'] = next_task / self.num_increments
        print(f"Switching task on all EnvRunners up to #{next_task}/{self.num_increments}; {self.env_config}")
        # Increase task.
        algorithm.env_runner_group.foreach_env_runner(
            func=partial(_remote_fn, config=self.env_config)
        )
        metrics_logger.log_value("current_env_task", next_task, window=1)

    def on_train_result(
        self,
        *,
        algorithm: Algorithm,
        metrics_logger=None,
        result: dict,
        **kwargs,
    ) -> None:
        # Store the current task inside the metrics logger in our Algorithm.
        current_task = metrics_logger.peek("current_env_task")
        if (current_task==self.num_increments):
            return # Gravity at maximum

        # Note, in the first callback executions there may be no completed episode
        # (and therefore no episode return) reported. In this case we will skip the
        # the logic to manage task difficulty.
        if EPISODE_RETURN_MEAN in result[ENV_RUNNER_RESULTS]:
            current_return = result[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]
        else:
            return

        # Check promotion (increasing task).
        if (current_return > self.promotion_threshold):
            cycles_waited = metrics_logger.peek("promotion_cycles")
            if (cycles_waited > self.promotion_patience):
                self.promote(algorithm, current_task, metrics_logger)
                metrics_logger.log_value("promotion_cycles", 0, window=1) # Reset the counter for next promotion
            else:
                metrics_logger.log_value("promotion_cycles", cycles_waited+1, window=1)
                print(f"Waiting until {self.promotion_patience} for promotion: {cycles_waited}")
        else:
            metrics_logger.log_value("promotion_cycles", 0, window=1) # Reset, it was a fluke.
                
            
                