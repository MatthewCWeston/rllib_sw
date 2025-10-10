import gymnasium as gym
import numpy as np
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.env.vector.vector_multi_agent_env import VectorMultiAgentEnv
from ray.rllib.utils.images import resize

class RenderCallback(RLlibCallback):
    """
        Rendering every single episode is a waste. Render the first episode only.
    """

    def __init__(self, render_every=1):
        super().__init__()
        # Only render and record on certain EnvRunner indices?
        self.render_every = render_every
        self.video = None
        self.sample_number = 0

    def on_episode_step(
        self,
        *,
        episode,
        env_runner,
        metrics_logger,
        env,
        env_index,
        rl_module,
        **kwargs,
    ) -> None:
        if ((self.sample_number%self.render_every != 0) or 
            (env_runner.worker_index != 1) or 
            (self.video is not None)):
            return # Only the first episode on the first runner

        # If we have a vector env, only render the sub-env at index 0.
        if isinstance(env.unwrapped, (gym.vector.VectorEnv, VectorMultiAgentEnv)):
            image = env.unwrapped.envs[0].render()
        # Render the gym.Env.
        else:
            image = env.unwrapped.render()
        image = np.array(image)
        # Resize to smaller image
        #image = resize(image, 128, 128)
        # For WandB videos, we need to put channels first.
        image = np.transpose(image, axes=[2, 0, 1])
        # Add the compiled single-step image as temp. data to our Episode object.
        # Once the episode is done, we'll compile the video from all logged images
        # and log the video with the EnvRunner's `MetricsLogger.log_...()` APIs.
        # See below:
        # `on_episode_end()`: We compile the video and maybe store it).
        # `on_sample_end()` We log the best and worst video to the `MetricsLogger`.
        if "render_images" not in episode.custom_data:
            episode.custom_data["render_images"] = []
        episode.custom_data["render_images"].append(image)
        

    def on_episode_end(
        self,
        *,
        episode,
        env_runner,
        metrics_logger,
        env,
        env_index,
        rl_module,
        **kwargs,
    ) -> None:
        if ((self.sample_number%self.render_every != 0) or 
            (env_runner.worker_index != 1) or 
            (self.video is not None)):
            return # Only the first episode on the first runner
        # Pull all images from the temp. data of the episode.
        images = episode.custom_data["render_images"]
        self.video = np.expand_dims(np.stack(images, axis=0), axis=0)

    def on_sample_end(
        self,
        *,
        env_runner,
        metrics_logger,
        samples,
        **kwargs,
    ) -> None:
        """Logs the video to this EnvRunner's MetricsLogger."""
        if self.video is not None:
            metrics_logger.log_value(
                "episode_videos",
                self.video,
                reduce=None,
                clear_on_reduce=True,
            )
            self.video = None
        self.sample_number += 1