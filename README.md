# Reinforcement Learning for `SPACEWAR!`

The aim of this repository is to demonstrate the subtleties of practical reinforcement learning in difficult settings that aren't covered in most tutorials, or, indeed, most lectures[^1]. To this ends, we attempt to solve an environment which involves multiple infamous RL bugbears:

 - Competitive, zero-sum multi-agent competition
 - Stochastic environment mechanics (Minskytron hyperspace)
 - Sparse rewards, occurring once, at the end of an episode.
 - Long episodes (800 steps in the settings used)
 - Variable-length observations composed of multiple types of object. *(And no, we won't cheat by feeding screenshots into a ConvNet - we're using a Transformer.)*
 
 Environment settings, to the best of my ability, have fidelity to the authentic `SPACEWAR!` implementation available at https://www.masswerk.at/spacewar/. I'd be honored if one of the greats would give my agent a shot.
 
 ## How to use this repository
 
 If you want to train a `SPACEWAR!` agent, run the following three commands in sequence:
 
 **Train a basic agent against an opponent that does nothing, but is placed in a stable orbit on environment start.** This allows it to learn the controls and the objective beforehand.
 ```
  python run_training.py --env-name SW_1v1_env_singleplayer --env-config '{"speed": 5.0, "ep_length": 4096, "size_multiplier": 1.0, "grav_multiplier": 1.0, "target_speed": 1.0, "target_ammo": 0.0, "aug_obs": true, "randomize_ammo": true, "no_respawn": false, "stochastic_hspace": true}' --verbose 1 --batch-size 65536 --minibatch-size 8192 --gamma .999 --attn-dim 128 --attn-ff-dim 1024 --lr 3e-5 --lambda_ .8 --vf-clip 40 --stop-iters=800 --num-env-runners 40 --envs-per-env-runner 8 --use-layernorm --activation-fn leakyrelu --checkpoint-freq 100 --checkpoint-at-end
 ```
 
 **Train the initial agent to respond to being fired upon.** This is, in essence, the full competitive environment, just with a fixed opponent.
 ```
 python run_training.py --env-name SW_1v1_env_singleplayer --env-config '{"speed": 5.0, "ep_length": 4096, "size_multiplier": 1.0, "grav_multiplier": 1.0, "target_speed": 1.0, "target_ammo": 1.0, "aug_obs": true, "randomize_ammo": true, "no_respawn": false, "stochastic_hspace": true}' --verbose 1 --batch-size 65536 --minibatch-size 8192 --gamma .999 --attn-dim 128 --attn-ff-dim 1024 --lr 3e-5 --lambda_ .8 --vf-clip 40 --stop-iters=700 --num-env-runners 40 --envs-per-env-runner 8 --use-layernorm --activation-fn leakyrelu --checkpoint-freq 100 --checkpoint-at-end --restore-checkpoint <path_to_your_final_stage_1_checkpoint_here, or use mine>
 ```
 
**Run PFSP.** This is, in essence, the full competitive environment. It includes intelligent checkpointing, using [Bradley-Terry](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model) score to evaluate agent skill.
 ```
 python run_training_MA.py --env-config '{"speed": 5.0, "ep_length": 4096, "aug_obs": true, "random_orbit_prob": 0.75, "stochastic_hspace": true}' --verbose 1 --batch-size 65536 --minibatch-size 8192 --gamma .999 --attn-dim 128 --attn-ff-dim 1024 --lr 3e-5 --lambda_ .8 --vf-clip 40 --stop-iters=4000 --num-env-runners 40 --envs-per-env-runner 8 --use-layernorm --activation-fn leakyrelu --pfsp --steps-to-clone 50 --add-v0 --restore-checkpoint <path_to_your_final_stage_2_checkpoint_here, or use mine> --checkpoint-freq 100 --checkpoint-at-end --identity-aug --iters-to-warmup-new 25
 ```
 
 **Evaluate your agents.** You can assess agents' Bradley-Terry ratings using `agent_comparison/BradleyTerry_rllib.py`. You can also visualize your agents' behavior, and play against them yourself, using `launch_game_MA.py`
 
 ```
 python -m agent_comparison.BradleyTerry_rllib --env-config '{"speed": 5.0, "ep_length": 4096, "aug_obs": true, "stochastic_hspace": true}' --attn-dim 128 --attn-ff-dim 1024  --use-layernorm --activation-fn leakyrelu --envs-per-env-runner 4 --evaluation-duration 1000 --eval-batch-size 200 --evaluation-num-env-runners 40 --agent-folder <folder_containing_agent_checkpoints_here>
 ```
 
 ```
 python launch_game_MA.py SW_1v1_env --env-config '{"speed": 5.0, "ep_length": 4096, "aug_obs": true, "stochastic_hspace": true}' --ckpt-path <path_to_your_pfsp_checkpoint_here>
 ```
 
 ## Some things I learned that worked
 
  - Neural network plasticity is absolutely vital here. Not only is it necessary to make the curriculum learning work, but the implicit curriculum that self-play algorithms create also requires good plasticity. It's a criminally underexplored research area, but I found that layer normalization and leaky ReLU layers are a substantial boon.
        - I note that, if you can't get your network to be sufficiently plastic, you can cold start your value function at each stage as a workaround. This isn't as elegant, though, so I don't like it as much.
    
  - This may seem obvious, but layer initialization matters a lot. PyTorch uses very different scales for linear layer weights and embedding weights, so be warned when you're training a transformer model that mixes the two. When performing model surgery of the kind employed by [OpenAI Five](https://arxiv.org/abs/1912.06680) *(the training pipeline above does this implicitly)*, you want to be very careful with your initialization values.
  
  - In self-play environments, random initialization makes all the difference in the world. Agents that start in the same positions every time will get stuck in local maxima, refining an all-in aggressive strategy and faltering against an opponent that does anything else. Placing the agents in opposing, stable orbits at environment start allows for much more efficient training of a well-rounded agent.
  
  - Curriculum learning is great, but don't overdo it. Stage 1 training resulted in much better performance in stage 2 than even a very generous stage-2-only training run, but a gradual shift in the environment's dynamics only confused earlier models. 
        - **Remember:** the point of curriculum learning is to make sure the agent always has an environment that's easy enough for winning to be a possibility. You don't need to micromanage it, especially when that involves changing the core mechanics.
        - A natural curriculum is ideal. For example, our single-agent environment repeatedly respawns the target, preserving the agent's location, heading, and resources, after the agent succeeds in hitting it. This proved crucial for teaching proper resource conservation.
   
  - Consider your skill evaluation algorithm. Many methods make affordances for human behavior that aren't needed here, and end up being counterproductive. Bradley-Terry is my personal favorite for ease of use and lack of assumptions. It'll always converge to a neat set of scores, while methods like Elo won't do so.
  
  - Pure self-play is well below optimal, at least when working with PPO agents. It's unstable; you want [PFSP](https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/), at the very least. Including a 50% pure self-play ratio, as AlphaStar did when no exploiters were ready for use, helped substantially with performance.
 
 ## Some things I implemented that didn't end up helping
 
  - Probabilistic critics seem like a naturally valuable thing, especially in stochastic environments, and doubly so when their plasticity benefits can be exploited. [EPPO](https://arxiv.org/abs/2503.01468) was an enjoyable paper to read, but I wasn't able to get stable performance out of it here. Have a look at my implementation, maybe you can make it work.
  
  - I tried a few gated transformer implementations, alongside other architectures and pooling mechanisms. The single attention layer inspired by [OpenAI's Hide and Seek paper](https://arxiv.org/abs/1909.07528) seems to work best. Remember that reinforcement learning loves smaller model sizes, by virtue of the fact that you're not getting the credit-assignment benefits of supervised learning.
  
  - Exploration incentives, from simple entropy coefficients to complex unsupervised curiosity incentives, vary between unnecessary and counterproductive in environments like this one, where entropy never drops very low without them. It's likely better to consider them a band-aid to be used if needed than as something that *must* be included in any respectable training run.
  
  - I adapted [autoresearch](https://github.com/MatthewCWeston/autoresearch_RL_transformer) to this problem, but ultimately didn't end up getting any successful experiments that ended up panning out in practice. Perhaps reinforcement learning is ill-suited to the kind of loop shown, especially because of the fact that what gets you the fastest performance jump in the first ten minutes isn't always what gets you the highest plateau an hour later.
 
 ## Future Work
 
 There's still plenty to be done. I've included three checkpoints in my release, one for each stage of training. 
 
  - **Can you design a better training regimen** that handily beats the final checkpoint *(without directly training on it, of course!)*? I've included scripts that let you see how an agent you trained is doing against other agents.
  
  - **What about team play?** The `CLS token` pooling method has performance comparable to mean pooling, and could be used to produce output logits for multiple ships on the same team. The environments and random state generation could be likewise extended. Can you train an agent that is able to work effectively alongside a human player to defeat a team of bots?
        - Consider the [OverCooked Generalization Challenge](https://arxiv.org/abs/2406.17949) paper and the papers that cite it. Generalizing to arbitrary teammates is often trickier than generalizing to arbitrary opponents, because conventional training doesn't inherently incentivize teammates to avoid each others' comfort zones!
    
  - **Is AlphaStar worth using?** I ran PFSP for efficiency's sake, but [AlphaStar](https://www.nature.com/articles/s41586-019-1724-z) does justify the value of an exploiter agent. [I've implemented AlphaStar in RLlib](https://github.com/MatthewCWeston/probabilistic_value_ppo/blob/main/RLlib_AlphaStar.ipynb), if you want to give it a shot here.
  
  - **Can you get something useful out of autoresearch?** LLMs are improving all the time, as is the infrastructure to use them, so if you're reading this in 2027, there's a good chance you can run my code out of the box and get better results.
        - Or maybe I was doing it wrong, and you've got a better prompt. Either way!
        
  - **I'm certain there's something to be learned about performance when agent identities are conveyed to the critic**. [MADDPG](https://arxiv.org/abs/1706.02275) famously discussed the value of stabilizing learning by providing other agents' policy information to the critic. I've implemented something *(very)* broadly similar under PPO in this repository, but I don't have the compute budget to robustly evaluate how helpful it is.
  
Certainly fork this repo if you want to run further experiments. I'd be glad to hear about any useful techniques that I missed in my explorations.

[^1]: [This lecture](https://www.youtube.com/watch?v=8EcdaCk9KaQ) is one of my very favorites for touching on the subject and covering a number of things I didn't know. Moreover, *"Don't fall into the trap of watching your algorithm spit out numbers all day"* is great advice. Run experiments in parallel, and learn proactively about hyperparameter search.