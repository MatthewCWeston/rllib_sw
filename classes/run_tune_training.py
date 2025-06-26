import os
import argparse
from typing import (TYPE_CHECKING,Any,Dict,List,Optional,Tuple,Type,Union,)
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.result import TRAINING_ITERATION
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)
# modify run_rllib_example_script_experiment in accordance with the other example, see if that gets us better results.
def get_tune_callbacks(args: argparse.Namespace):
    if (args.wandb_key is not None):
        return[WandbLoggerCallback(
                api_key=args.wandb_key,
                project=args.wandb_project,
                upload_checkpoints=True,
                **({"name": args.wandb_run_name} if args.wandb_run_name else {}),
        )]
    return []

def run_tune_training(config: "AlgorithmConfig",args: Optional[argparse.Namespace] = None,*,stop: Optional[Dict] = None, scheduler=None):
    # Initialize Ray.
    ray.init(num_cpus=args.num_cpus or None, local_mode=args.local_mode, ignore_reinit_error=True,)
    config.learners(num_gpus_per_learner=1)
    # Auto-configure a CLIReporter (to log the results to the console).
    # Use better ProgressReporter for multi-agent cases: List individual policy rewards.
    progress_reporter = CLIReporter(
        metric_columns={
            **{
                TRAINING_ITERATION: "iter",
                "time_total_s": "total time (s)",
                NUM_ENV_STEPS_SAMPLED_LIFETIME: "ts",
                f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": "combined return",
            },
            **{
                (
                    f"{ENV_RUNNER_RESULTS}/module_episode_returns_mean/" f"{pid}"
                ): f"return {pid}"
                for pid in config.policies
            },
        },
    )
    os.environ["RAY_AIR_NEW_OUTPUT"] = "0"
    # Run the actual experiment (using Tune).
    tune.Tuner(
        config.algo_class,
        param_space=config,
        run_config=tune.RunConfig(
            stop=stop,
            verbose=args.verbose,
            callbacks=get_tune_callbacks(args),
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_frequency=args.checkpoint_freq,
                checkpoint_at_end=args.checkpoint_at_end,
            ),
            progress_reporter=progress_reporter,
        ),
        tune_config=tune.TuneConfig(
            num_samples=args.num_samples,
            max_concurrent_trials=args.max_concurrent_trials,
            scheduler=scheduler,
        ),
    ).fit()
    ray.shutdown()