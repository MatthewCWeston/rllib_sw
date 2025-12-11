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
        attributes: list,
        promotion_thresholds: dict, # name -> (results_key, threshold_value)
        promotion_patience: int = 2,
        num_increments: int = 10,
        allow_demotions: bool = False,
    ):
        self.env_config = env_config.copy() # Adjust gravity here
        self.promotion_thresholds = promotion_thresholds
        self.promotion_patience = promotion_patience
        self.num_increments = num_increments
        self.allow_demotions = allow_demotions
        # name -> (start, increment)
        ATTRS_FINAL = {"grav_multiplier": 1.0, "size_multiplier": 1.0, "target_speed": 1.0,
            "target_ammo": 1.0}
        self.attribute_dict = { # Start from [0], increment by [1] at each step
            n: (env_config[n], (ATTRS_FINAL[n]-env_config[n])/num_increments) for n in attributes
        }
        print("Curriculum learning:")
        print(self.attribute_dict)
        
    def on_algorithm_init(
        self,
        *,
        algorithm: Algorithm,
        **kwargs,
    ) -> None:
        # Set the initial task to 0.
        for k in self.attribute_dict.keys():
            algorithm.metrics.log_value(f"{k}_current_env_task", 0, window=1)
            algorithm.metrics.log_value(f"{k}_promotion_cycles", 0, window=1)
            algorithm.metrics.log_value(f"{k}_demotion_cycles", 0, window=1)
        
    def promote(self, k, algorithm, current_task, metrics_logger, adj=1.0):
        next_task = current_task + adj
        base, increment = self.attribute_dict[k]
        self.env_config[k] = base + next_task*increment
        print(f"Switching {k} task on all EnvRunners up to {self.env_config[k]:.2f}; {self.env_config}")
        metrics_logger.log_value(f"{k}_current_env_task", next_task, window=1)

    def on_train_result(
        self,
        *,
        algorithm: Algorithm,
        metrics_logger=None,
        result: dict,
        **kwargs,
    ) -> None:
        promoted = False
        for k in self.attribute_dict.keys():
            p_cycles = f"{k}_promotion_cycles"
            d_cycles = f"{k}_demotion_cycles"
            pt_key, pt_val = self.promotion_thresholds[k]
            # Store the current task inside the metrics logger in our Algorithm.
            current_task = metrics_logger.peek(f"{k}_current_env_task")
            if (current_task==self.num_increments):
                return # Difficulty at maximum

            # Note, in the first callback executions there may be no completed episode
            # (and therefore no episode return) reported. In this case we will skip the
            # the logic to manage task difficulty.
            if pt_key in result[ENV_RUNNER_RESULTS]:
                current_return = result[ENV_RUNNER_RESULTS][pt_key]
            else:
                return

            # Check promotion (increasing task).
            if (current_return >= pt_val):
                cycles_waited = metrics_logger.peek(p_cycles)
                if (cycles_waited > self.promotion_patience):
                    promoted = True
                    self.promote(k, algorithm, current_task, metrics_logger)
                    metrics_logger.log_value(p_cycles, 0, window=1) # Reset the counter for next promotion
                else:
                    metrics_logger.log_value(p_cycles, cycles_waited+1, window=1)
                    print(f"{k}: Waiting until {self.promotion_patience} for promotion: {cycles_waited}")
            else:
                metrics_logger.log_value(p_cycles, 0, window=1) # Reset, it was a fluke.
            # Check demotion, if allowed
            if (self.allow_demotions):
                if (current_return < pt_val):
                    if (current_task > 0): # Check demotion if possible to demote 
                        cycles_waited = metrics_logger.peek(d_cycles)
                        if (cycles_waited > self.promotion_patience):
                            promoted = True
                            self.promote(k, algorithm, current_task, metrics_logger, adj=-1.0) # Go back down
                            metrics_logger.log_value(d_cycles, 0, window=1) # Reset the counter for next promotion
                        else:
                            metrics_logger.log_value(d_cycles, cycles_waited+1, window=1)
                            print(f"{k}: Waiting until {self.promotion_patience} for demotion: {cycles_waited}")
                else:
                    metrics_logger.log_value(d_cycles, 0, window=1) # Reset demotion cycles
        if (promoted):
            # Send updated config to envrunners
            algorithm.env_runner_group.foreach_env_runner(
                func=partial(_remote_fn, config=self.env_config)
            )
            
                