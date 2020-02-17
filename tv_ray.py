import ray
from ray.tune import run
from ray.tune import run_experiments, grid_search
from ray.tune.registry import register_env
from environment import Environment
import pdb
import itertools
from ray.tune.schedulers import AsyncHyperBandScheduler
import numpy as np
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt import hp
import argparse
import collections
import json
import os
import pickle
import pdb
import gym
import ray
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.evaluation.sample_batch import DEFAULT_POLICY_ID
from ray.tune.util import merge_dicts
from ray.tune.registry import register_env
from environment import Environment

TRAIN_TIMESTEPS = 10000
NUM_WORKERS = 3 #0 in remote #0 in local dqn # 3 in ppo
NUM_CPUS = 4 #1 in remote #1 in local dqn # 3 in ppo
NUM_GPUS = 0 #1 in remote #0 in local
ENV_NAME = "Projectlife-v1"
PAIR_LIST = ["XRPBTC"]#["DLTBTC","XLMBTC","LTCBTC","XMRBTC","BTCUSDT"]
INTERVAL_LIST = ["1h"]#["5m","15m","1h","4h","1d"]
ALGO_LIST = ["PPO"]#RAINBOWCUSTOM ["RAINBOW","PPO",A2C","A3C","DDPG","IMPALA","ARS"]
FEATURE_LIST = [["close","sma15","sma50"]]#[["close","sma20"],["close","sma50"],["close","rsi","adx"],["rsi","adx","atr"],["sma10","sma20","sma50"],["sma10"],["sma20"],["sma50"],["rsi"]]

def train():
	for pair, interval, algo, features in itertools.product(PAIR_LIST,INTERVAL_LIST,ALGO_LIST,FEATURE_LIST):
		experiment = ENV_NAME+"-"+pair+"-"+interval+"-"+algo+"-"+('-'.join(features))
		register_env(ENV_NAME, lambda config: Environment(mode="train", pair=pair, interval=interval,  algo=algo, data_features=features))
		ray.init(num_cpus=NUM_CPUS, num_gpus=NUM_GPUS)
		run(search_type="gridsearch", algo=algo, exp_name=experiment)

def run(search_type, algo,exp_name):
	if search_type=="gridsearch":
		run_experiments({exp_name: get_params(algo)})
	elif search_type=="hyperopt":
		space = {
			"double_q": hp.choice('double_q',[True, False]),
			"hiddens": [hp.choice('hiddens',[64,256,512,1024])],
			"buffer_size": hp.choice('buffer_size', np.arange(20000,100000,1000, dtype=int)),
			"lr": hp.loguniform("lr",np.log(0.0005),np.log(0.1)),
			"n_step":hp.choice('n_step',[1,3,6,12]),
			"target_network_update_freq": hp.choice('target_network_update_freq',[10, 30, 50])
		}
		search_alg = HyperOptSearch(space, max_concurrent=4,reward_attr="episode_reward_mean")
		scheduler = AsyncHyperBandScheduler(reward_attr="episode_reward_mean")
		run_experiments({exp_name: get_params(algo)}, search_alg=search_alg, scheduler=scheduler, verbose=True)

def get_params(algo):
	if algo=="RAINBOWCUSTOM":
		params = {
					"run": "DQN",
					"env": ENV_NAME,
					"stop": {
						"timesteps_total": TRAIN_TIMESTEPS,
					},
					"checkpoint_freq": 100,
					"checkpoint_at_end": True,
					#"num_samples": 100,
					"config": {
								# === Running ===
								# Number of workers for collecting samples with. This only makes sense
								# to increase if your environment is particularly slow to sample, or if
								# you"re using the Async or Ape-X optimizers.
								"num_workers": NUM_WORKERS,
								"num_gpus": NUM_GPUS,
								# Optimizer class to use.
								"optimizer_class": "SyncReplayOptimizer",
								# Whether to use a distribution of epsilons across workers for exploration.
								"per_worker_exploration": False,
								# Whether to compute priorities on workers.
								"worker_side_prioritization": False,
								# Prevent iterations from going lower than this time span
								"min_iter_time_s": 1,

								# === Model ===
								# Number of atoms for representing the distribution of return. When
								# this is greater than 1, distributional Q-learning is used.
								# the discrete supports are bounded by v_min and v_max
								"num_atoms": 51,
								"v_min":-10, #grid_search([-400, -10]),
								"v_max": 10, #grid_search([400, 10]),
								# Whether to use noisy network
								"noisy": True,
								# control the initial value of noisy nets
								"gamma": 0.99,
								# Whether to use dueling dqn
								"dueling": True,
								# Whether to use double dqn
								"double_q": True, #grid_search([True, False]),
								# Postprocess model outputs with these hidden layers to compute the
								# state and action values. See also the model config in catalog.py.
								"hiddens":[64],#[grid_search([32,48])],#[grid_search([16,64,256,512,1024])],#[512],
								# N-step Q learning
								"n_step": 3, #grid_search([1,3,6]),
								"model": {
								    # == LSTM ==
								    # Whether to wrap the model with a LSTM
								    "use_lstm": True,
								    "lstm_cell_size": 256,
									"grayscale": True,
									"zero_mean": False,
									"dim": 42
								},
								# === Exploration ===
								# Max num timesteps for annealing schedules. Exploration is annealed from
								# 1.0 to exploration_fraction over this number of timesteps scaled by
								# exploration_fraction
								"schedule_max_timesteps": 2000000,
								# Number of env steps to optimize for before returning
								"timesteps_per_iteration": 1000,
								# Fraction of entire training period over which the exploration rate is
								# annealed
								"exploration_fraction": 0.000001,
								# Final value of random action probability
								"exploration_final_eps": 0,
								# Update the target network every `target_network_update_freq` steps.
								"target_network_update_freq": 30, #grid_search([10, 30, 50])
								# Use softmax for sampling actions. Required for off policy estimation.
								"soft_q": False,
								# Softmax temperature. Q values are divided by this value prior to softmax.
								# Softmax approaches argmax as the temperature drops to zero.
								"softmax_temp": 1.0,
								# If True parameter space noise will be used for exploration
								# See https://blog.openai.com/better-exploration-with-parameter-noise/
								"parameter_noise": True,

								# === Replay buffer ===
								# Size of the replay buffer. Note that if async_updates is set, then
								# each worker will have a replay buffer of this size.
								"buffer_size":50000, #grid_search([10000, 25000, 50000, 75000, 100000]),
								# If True prioritized replay buffer will be used.
								"prioritized_replay": True,
								# Alpha parameter for prioritized replay buffer.
								"prioritized_replay_alpha": 0.6,
								# Beta parameter for sampling from prioritized replay buffer.
								"prioritized_replay_beta": 0.4,
								# Fraction of entire training period over which the beta parameter is
								# annealed
								"beta_annealing_fraction": 0.2,
								# Final value of beta
								"final_prioritized_replay_beta": 1, #grid_search([0.1, 0.4,1]),
								# Epsilon to add to the TD errors when updating priorities.
								"prioritized_replay_eps": 1e-6,
								# Whether to LZ4 compress observations
								"compress_observations": True,
								# === Optimization ===
								# Learning rate for adam optimizer
								"lr": 0.0013,
								# Learning rate schedule
								"lr_schedule": None,
								# Adam epsilon hyper parameter
								"adam_epsilon": 1e-8,
								# If not None, clip gradients during optimization at this value
								"grad_norm_clipping": 40,
								# How many steps of the model to sample before learning starts.
								"learning_starts": 0,
								# Update the replay buffer with this many samples at once. Note that
								# this setting applies per-worker if num_workers > 1.
								"sample_batch_size": 4,
								# Size of a batched sampled from replay buffer for training. Note that
								# if async_updates is set, then each worker returns gradients for a
								# batch of this size.
								"train_batch_size": 32
					}
				}
	elif algo=="RAINBOW":
		params = {
					"run": "DQN",
					"env": ENV_NAME,
					"stop": {
						"timesteps_total": TRAIN_TIMESTEPS,
					},
					"checkpoint_freq": 100,
					"checkpoint_at_end": True,
					"config": {
						"num_atoms": 51,
						"noisy": True,
						"gamma": 0.99,
						"lr": 0.0001,
						"hiddens": [
							512
						],
						"learning_starts": 10,
						"buffer_size": 50000,
						"sample_batch_size": 4,
						"train_batch_size": 32,
						"schedule_max_timesteps": 2000000,
						"exploration_final_eps": 0,
						"exploration_fraction": 0.000001,
						"target_network_update_freq": 500,
						"prioritized_replay": True,
						"prioritized_replay_alpha": 0.5,
						"beta_annealing_fraction": 0.2,
						"final_prioritized_replay_beta": 1,
						"num_workers": NUM_WORKERS,
						"n_step": 3,
						"model": {
							"conv_filters":None,
							"use_lstm": True,
							"lstm_cell_size": 256
						},
						"num_gpus": NUM_GPUS
					}
				}
	elif algo=="PPO":
		params = {
					"run": algo,
					"env": ENV_NAME,
					"stop": {
						"timesteps_total": TRAIN_TIMESTEPS,
					},
					"checkpoint_freq": 100,
					"checkpoint_at_end": True,
					"config": {
						"train_batch_size": 2048,
						"vf_clip_param": 10,
						"num_workers": NUM_WORKERS,
						"lambda": 0.1,
						"gamma": 0.95,
						"lr": 0.0003,
						"sgd_minibatch_size": 64,
						"num_sgd_iter": 10,
						"model": {
							"use_lstm": True
						},
						"batch_mode": "complete_episodes",
						"observation_filter": "MeanStdFilter",
						"num_gpus": NUM_GPUS
					}
				}
	elif algo=="A2C":
		params = {
					"run": algo,
					"env": ENV_NAME,
					"stop": {
						"timesteps_total": TRAIN_TIMESTEPS,
					},
					"checkpoint_freq": 100,
					"checkpoint_at_end": True,
					"config": {
						"sample_batch_size": 20,
						"clip_rewards": True,
						"num_workers": NUM_WORKERS,
						"num_gpus": NUM_GPUS,
						"lr_schedule": [
							[
								0,
								0.0007
							],
							[
								20000000,
								1e-12
							]
						]
					}

				}
	elif algo=="A3C":
		params = {
					"run": algo,
					"env": ENV_NAME,
					"stop": {
						"timesteps_total": TRAIN_TIMESTEPS,
					},
					"checkpoint_freq": 100,
					"checkpoint_at_end": True,
					"config": {
						"num_workers": NUM_WORKERS,
						"num_gpus": NUM_GPUS,
			            "model": {
			                "conv_filters": [
			                    [16, [8, 8], 4],
			                    [32, [4, 4], 2],
			                    [512, [10, 10], 1],
			                ],
			            },
			            "gamma": 0.95
					}

				}
	elif algo=="DDPG":
		params = {
					"run": algo,
					"env": ENV_NAME,
					"stop": {
						"timesteps_total": TRAIN_TIMESTEPS,
					},
					"checkpoint_freq": 100,
					"checkpoint_at_end": True,
					"config": {
						"actor_hiddens": [
							64,
							64
						],
						"critic_hiddens": [
							64,
							64
						],
						"n_step": 1,
						"model": {},
						"gamma": 0.99,
						"env_config": {},
						"schedule_max_timesteps": 100000,
						"timesteps_per_iteration": 600,
						"exploration_fraction": 0.1,
						"exploration_final_eps": 0.02,
						"noise_scale": 0.1,
						"exploration_theta": 0.15,
						"exploration_sigma": 0.2,
						"target_network_update_freq": 0,
						"tau": 0.001,
						"buffer_size": 10000,
						"prioritized_replay": True,
						"prioritized_replay_alpha": 0.6,
						"prioritized_replay_beta": 0.4,
						"prioritized_replay_eps": 0.000001,
						"clip_rewards": False,
						"lr": grid_search([0.0001,0.0002]),
						"actor_loss_coeff": 0.1,
						"critic_loss_coeff": 1,
						"use_huber": True,
						"huber_threshold": 1,
						"l2_reg": 0.000001,
						"sample_batch_size": 1,
						"train_batch_size": 64,
						"num_workers": NUM_WORKERS,
						"optimizer_class": "SyncReplayOptimizer",
						"per_worker_exploration": False,
						"worker_side_prioritization": False,
						"num_gpus": NUM_GPUS
					}
				}
	elif algo=="IMPALA":
		params = {
					"run": algo,
					"env": ENV_NAME,
					"stop": {
						"timesteps_total": TRAIN_TIMESTEPS,
					},
					"checkpoint_freq": 100,
					"checkpoint_at_end": True,
					"config": {
						"sample_batch_size": 50,
						"train_batch_size": 500,
						"num_workers": NUM_WORKERS,
						"num_gpus": NUM_GPUS
					}
				}
	elif algo=="ARS":
		params = {
					"run": algo,
					"env": ENV_NAME,
					"stop": {
						"timesteps_total": TRAIN_TIMESTEPS,
					},
					"checkpoint_freq": 100,
					"checkpoint_at_end": True,
					"config": {
						"noise_stdev": 0.01,
						"num_rollouts": 1,
						"rollouts_used": 1,
						"num_workers": 1,
						"sgd_stepsize": 0.02,
						"eval_prob": 0.2,
						"offset": 0,
						"observation_filter":"NoFilter",
						"report_length": 3,
						"num_workers": NUM_WORKERS,
						"num_gpus": NUM_GPUS
					}
				}

	return params



def validate():
    pair = "XRPBTC"
    interval ="1h"
    algo = "ARS"
    features = ["close","sma15","sma50"]
    ENV_NAME = "Projectlife-v1"
    register_env(ENV_NAME, lambda config: Environment(mode="validate", pair=pair, interval=interval, algo=algo, data_features=features))
    parser = create_parser()
    args = parser.parse_args()
    config = {}
    config_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(config_dir, "params.pkl")
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")
    if not os.path.exists(config_path):
        if not args.config:
            raise ValueError(
                "Could not find params.pkl in either the checkpoint dir or "
                "its parent directory.")
    else:
        with open(config_path, "rb") as f:
            config = pickle.load(f)
    config = merge_dicts(config, args.config)
    ray.init()
    if algo == "RAINBOW":
        algo = "DQN"
    cls = get_agent_class(algo)
    agent = cls(env=ENV_NAME, config=config)
    agent.restore(args.checkpoint)
    policy_agent_mapping = default_policy_agent_mapping
    if hasattr(agent, "local_evaluator"):
        env = agent.local_evaluator.env
        multiagent = isinstance(env, MultiAgentEnv)
        if agent.local_evaluator.multiagent:
            policy_agent_mapping = agent.config["multiagent"][
                "policy_mapping_fn"]

        policy_map = agent.local_evaluator.policy_map
        state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
        use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
        action_init = {
            p: m.action_space.sample()
            for p, m in policy_map.items()
        }
    else:
        env = gym.make(ENV_NAME)
        multiagent = False
        use_lstm = {DEFAULT_POLICY_ID: False}

    steps = 0
    while steps < (len(env.df) or steps + 1):
        mapping_cache = {}  # in case policy_agent_mapping is stochastic
        obs = env.reset()
        agent_states = DefaultMapping(
            lambda agent_id: state_init[mapping_cache[agent_id]])
        prev_actions = DefaultMapping(
            lambda agent_id: action_init[mapping_cache[agent_id]])
        prev_rewards = collections.defaultdict(lambda: 0.)
        done = False
        reward_total = 0.0
        while not done and steps < (len(env.df) or steps + 1):
            multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}
            action_dict = {}
            for agent_id, a_obs in multi_obs.items():
                if a_obs is not None:
                    policy_id = mapping_cache.setdefault(
                        agent_id, policy_agent_mapping(agent_id))
                    p_use_lstm = use_lstm[policy_id]
                    if p_use_lstm:
                        a_action, p_state, _ = agent.compute_action(
                            a_obs,
                            state=agent_states[agent_id],
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                        agent_states[agent_id] = p_state
                    else:
                        a_action = agent.compute_action(
                            a_obs,
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                    action_dict[agent_id] = a_action
                    prev_actions[agent_id] = a_action
            action = action_dict

            action = action if multiagent else action[_DUMMY_AGENT_ID]
            next_obs, reward, done, _ = env.step(action)
            if multiagent:
                for agent_id, r in reward.items():
                    prev_rewards[agent_id] = r
            else:
                prev_rewards[_DUMMY_AGENT_ID] = reward

            if multiagent:
                done = done["__all__"]
                reward_total += sum(reward.values())
            else:
                reward_total += reward
            env.render()
            steps += 1
            obs = next_obs
        print("Episode reward", reward_total)

def create_parser(parser_creator=None):
    parser_creator = parser_creator or argparse.ArgumentParser
    parser = parser_creator(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Roll out a reinforcement learning agent "
        "given a checkpoint.")

    parser.add_argument(
        "checkpoint", type=str, help="Checkpoint from which to roll out.")
    required_named = parser.add_argument_group("required named arguments")
    parser.add_argument(
        "--config",
        default="{}",
        type=json.loads,
        help="Algorithm-specific configuration (e.g. env, hyperparams). "
        "Surpresses loading of configuration from checkpoint.")
    return parser


class DefaultMapping(collections.defaultdict):
    """default_factory now takes as an argument the missing key."""

    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value


def default_policy_agent_mapping(unused_agent_id):
    return DEFAULT_POLICY_ID


if __name__ == '__main__':
	train()
	#validate()




