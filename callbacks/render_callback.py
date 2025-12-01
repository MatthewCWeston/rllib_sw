import gymnasium as gym
import numpy as np
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.env.vector.vector_multi_agent_env import VectorMultiAgentEnv
from ray.rllib.utils.images import resize

class RenderCallback(RLlibCallback):
    """
        Rendering every single episode is a waste. Render the first episode only.
         - Remember to set render_size to something reasonable!
    """
    
    # python /mnt/c/Users/USER/Downloads/ray/rllib/examples/envs/env_rendering_and_recording.py --env=CartPole-v1 --stop-iters=2

    def __init__(self, render_every=1):
        super().__init__()
        # Only render and record on certain EnvRunner indices?
        self.render_every = render_every
        self.recorded_episode = None
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
            ((episode.id_ != self.recorded_episode) and (self.recorded_episode is not None))):
            return # Only the first episode on the first runner
        # If we have a vector env, only render the sub-env at index 0.
        if isinstance(env.unwrapped, (gym.vector.VectorEnv, VectorMultiAgentEnv)):
            image = env.unwrapped.envs[0].render()
        # Render the gym.Env.
        else:
            image = env.unwrapped.render()
        image = np.array(image, dtype=np.uint8) # Remember not to save a giant image of floating points!
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
            self.recorded_episode = episode.id_
        #print(f"\t === Episode {episode.id_} ongoing, {len(episode.custom_data['render_images'])} images recorded, worker is {env_runner.worker_index}")
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
        if ((self.sample_number%self.render_every != 0) or  # No video this epoch
            (env_runner.worker_index != 1) or               # We only want a single worker
            (self.recorded_episode is None) or              # Can't log a video we don't have
            (episode.id_ != self.recorded_episode)):        # Can't log an episode we haven't recorded
            return # Only the first episode on the first runner
        #print(f"\t\t ==== Episode {episode.id_} end, sample number is {self.sample_number} worker is {env_runner.worker_index}")
        # Pull all images from the temp. data of the episode.
        images = episode.custom_data["render_images"]
        video = np.expand_dims(np.stack(images, axis=0, dtype=np.uint8), axis=0)
        # Log the video here instead
        #print("\t\t\t ===== Logging video")
        metrics_logger.log_value(
            "episode_videos",
            video,
            reduce=None,
            clear_on_reduce=True,
        )
        print(len(metrics_logger.peek("episode_videos")))
        self.video = None

    def on_sample_end(
        self,
        *,
        env_runner,
        metrics_logger,
        samples,
        **kwargs,
    ) -> None:
        self.sample_number += 1
        self.recorded_episode = None