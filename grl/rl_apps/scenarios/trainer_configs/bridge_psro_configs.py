import os
from typing import Dict, Any

from ray.rllib.env import MultiAgentEnv
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.models.torch.torch_action_dist import TorchBeta
from ray.rllib.utils import merge_dicts
from ray.tune.registry import RLLIB_ACTION_DIST, _global_registry

from grl.rl_apps.scenarios.trainer_configs.defaults import GRL_DEFAULT_OPENSPIEL_POKER_DQN_PARAMS, \
    GRL_DEFAULT_POKER_PPO_PARAMS
from grl.rllib_tools.action_dists import TorchGaussianSquashedGaussian
from grl.rllib_tools.models.valid_actions_fcnet import get_valid_action_fcn_class_for_env
from grl.rllib_tools.valid_actions_epsilon_greedy import ValidActionsEpsilonGreedy

# from grl.envs.tiny_bridge_2p_multi_agent_env  import TinyBridge2pMultiAgentEnv
from grl.envs.tiny_bridge_4p_multi_agent_env  import TinyBridge4pMultiAgentEnv
from grl.envs.poker_4p_multi_agent_env import Poker4PMultiAgentEnv
from grl.rllib_tools.models.valid_actions_fcnet import get_valid_action_fcn_class_for_env
from grl.rl_apps.centralized_critic_model_kuhn import TorchCentralizedCriticModel
from grl.rl_apps.centralized_critic_model_full_obs_larger_network_kuhn import TorchCentralizedCriticModelFullObsLargerModelKuhn
from grl.rl_apps.kuhn_4p_mappo_full_obs_larger import kuhn_CCTrainer_4P_full_obs_larger, kuhn_CCPPOTorchPolicy_4P_full_obs_larger
from gym.spaces import Discrete

from ray.rllib.models import ModelCatalog

ModelCatalog.register_custom_model(
        "cc_model", TorchCentralizedCriticModel)

def team_psro_kuhm_ccppo_params_larger(env: MultiAgentEnv) -> Dict[str, Any]:
    env_config={
        "version": "kuhn_4p",
        "fixed_players": True,
    }
    tmp_env = Poker4PMultiAgentEnv(env_config)
    config = {
        "clip_param": 0.03,
        "entropy_coeff": 0.00,
        "framework": "torch",
        "gamma": 1.0,
        "kl_coeff": 0.2,
        "kl_target": 0.001,
        "lr": 5e-4,
        "metrics_smoothing_episodes": 5000,
        "model": {
            "custom_model": "cc_model_full_obs_larger",
            "vf_share_layers": False
        },
        "batch_mode": "complete_episodes",
        "num_envs_per_worker": 1,
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_gpus_per_worker": 0.0,
        "num_sgd_iter": 10,
        "rollout_fragment_length": 256,
        "sgd_minibatch_size": 256,
        "train_batch_size": 4096,
        "vf_clip_param": 5.0,
        "vf_share_layers": False,
        "framework": "torch",
        "num_workers": 25,

    }
    return merge_dicts(GRL_DEFAULT_POKER_PPO_PARAMS, config)

def psro_tiny_bridge_ccppo_params(env: MultiAgentEnv) -> Dict[str, Any]:
    env_config={
        "version": "tiny_bridge_4p",
        "fixed_players": True,
    }
    tmp_env = TinyBridge4pMultiAgentEnv(env_config)
    config = {
        "clip_param": 0.03,
        "entropy_coeff": 0.00,
        "framework": "torch",
        "gamma": 1.0,
        "kl_coeff": 0.2,
        "kl_target": 0.001,
        "lr": 5e-4,
        "metrics_smoothing_episodes": 5000,
        "model": {
            "custom_model": "cc_model_full_obs",
            "vf_share_layers": False
        },
        "batch_mode": "complete_episodes",
        "num_envs_per_worker": 1,
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_gpus_per_worker": 0.0,
        "num_sgd_iter": 10,
        "rollout_fragment_length": 256,
        "sgd_minibatch_size": 256,
        "train_batch_size": 4096,
        "vf_clip_param": 5.0,
        "vf_share_layers": False,
        "framework": "torch",
        "num_workers": 25,

    }
    return merge_dicts(GRL_DEFAULT_POKER_PPO_PARAMS, config)


def psro_tiny_bridge_ccppo_params_larger(env: MultiAgentEnv) -> Dict[str, Any]:
    env_config={
        "version": "tiny_bridge_4p",
        "fixed_players": True,
    }
    tmp_env = TinyBridge4pMultiAgentEnv(env_config)
    config = {
        "clip_param": 0.03,
        "entropy_coeff": 0.00,
        "framework": "torch",
        "gamma": 1.0,
        "kl_coeff": 0.2,
        "kl_target": 0.01,
        "lr": 5e-4,
        "metrics_smoothing_episodes": 5000,
        "model": {
            "custom_model": "cc_model_full_obs_larger_tiny_bridge_4p",
            "vf_share_layers": False
        },
        "batch_mode": "complete_episodes",
        "num_envs_per_worker": 1,
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_gpus_per_worker": 0.0,
        "num_sgd_iter": 5,
        "rollout_fragment_length": 256,
        "sgd_minibatch_size": 128,
        "train_batch_size": 2048,
        "vf_clip_param": 1.0,
        "vf_share_layers": False,
        "framework": "torch",
        "num_workers": 25,
    }
    return merge_dicts(GRL_DEFAULT_POKER_PPO_PARAMS, config)

def psro_tiny_bridge_ccppo_params_indep(env: MultiAgentEnv) -> Dict[str, Any]:
    env_config={
        "version": "tiny_bridge_4p",
        "fixed_players": True,
    }
    tmp_env = TinyBridge4pMultiAgentEnv(env_config)
    config = {
        "clip_param": 0.03,
        "entropy_coeff": 0.00,
        "framework": "torch",
        "gamma": 1.0,
        "kl_coeff": 0.2,
        "kl_target": 0.001,
        "lr": 5e-5,
        "model": {
            "custom_model": "cc_model_full_obs_larger_tiny_bridge_4p",
            "vf_share_layers": False
        },
        "metrics_smoothing_episodes": 5000,
        "batch_mode": "complete_episodes",
        "num_envs_per_worker": 1,
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_gpus_per_worker": 0.0,
        "num_sgd_iter": 10,
        "rollout_fragment_length": 256,
        "sgd_minibatch_size": 256,
        "train_batch_size": 4096,
        "vf_clip_param": 5.0,
        "vf_share_layers": False,
        "framework": "torch",
        "num_workers": 5,

    }
    return merge_dicts(GRL_DEFAULT_POKER_PPO_PARAMS, config)


def psro_bridge_ccppo_params(env: MultiAgentEnv) -> Dict[str, Any]:
    env_config={
        "version": "bridge",
        "fixed_players": True,
    }
    config = {
        "clip_param": 0.03,
        "entropy_coeff": 0.00,
        "framework": "torch",
        "gamma": 1.0,
        "kl_coeff": 0.2,
        "kl_target": 0.001,
        "lr": 5e-5,
        "metrics_smoothing_episodes": 5000,
        "model": {
            "custom_model": "cc_model_full_obs",
            "vf_share_layers": False
        },
        "batch_mode": "complete_episodes",
        "num_envs_per_worker": 1,
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_gpus_per_worker": 0.0,
        "num_sgd_iter": 10,
        "rollout_fragment_length": 256,
        "sgd_minibatch_size": 256,
        "train_batch_size": 4096,
        "vf_clip_param": 5.0,
        "vf_share_layers": False,
        "framework": "torch",
        "num_workers": 25,

    }
    return merge_dicts(GRL_DEFAULT_POKER_PPO_PARAMS, config)


def psro_bridge_ccppo_params_larger(env: MultiAgentEnv) -> Dict[str, Any]:
    env_config={
        "version": "bridge",
        "fixed_players": True,
    }
    config = {
        "clip_param": 0.03,
        "entropy_coeff": 0.0001,
        "framework": "torch",
        "gamma": 1.0,
        "kl_coeff": 0.2,
        "kl_target": 0.01,
        "lr": 5e-4,
        "metrics_smoothing_episodes": 5000,
        "model": {
            "custom_model": "cc_model_full_obs_larger_bridge",
            "vf_share_layers": False,
            "fcnet_hiddens": [256, 128],
        },
        "batch_mode": "complete_episodes",
        "num_envs_per_worker": 1,
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_gpus_per_worker": 0.0,
        "num_sgd_iter": 5,
        "rollout_fragment_length": 256,
        "sgd_minibatch_size": 128,
        "train_batch_size": 4096,
        "vf_clip_param": 1.0,
        "vf_share_layers": False,
        "framework": "torch",
        "num_workers": 25,
    }
    return merge_dicts(GRL_DEFAULT_POKER_PPO_PARAMS, config)

def psro_bridge_ccppo_params_indep(env: MultiAgentEnv) -> Dict[str, Any]:
    env_config={
        "version": "bridge",
        "fixed_players": True,
    }
    config = {
        "clip_param": 0.03,
        "entropy_coeff": 0.00,
        "framework": "torch",
        "gamma": 1.0,
        "kl_coeff": 0.2,
        "kl_target": 0.001,
        "lr": 5e-5,
        "metrics_smoothing_episodes": 5000,
        "model": {
            "custom_model": "cc_model_full_obs_larger_bridge",
            "vf_share_layers": False,
            "fcnet_hiddens": [256, 128],
        },
        "batch_mode": "complete_episodes",
        "num_envs_per_worker": 1,
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_gpus_per_worker": 0.0,
        "num_sgd_iter": 10,
        "rollout_fragment_length": 256,
        "sgd_minibatch_size": 256,
        "train_batch_size": 4096,
        "vf_clip_param": 5.0,
        "vf_share_layers": False,
        "framework": "torch",
        "num_workers": 5,
    }
    return merge_dicts(GRL_DEFAULT_POKER_PPO_PARAMS, config)


def psro_full_bridge_ccppo_params(env: MultiAgentEnv) -> Dict[str, Any]:
    env_config={
        "version": "bridge",
        "fixed_players": True,
    }
    config = {
        "clip_param": 0.03,
        "entropy_coeff": 0.00,
        "framework": "torch",
        "gamma": 1.0,
        "kl_coeff": 0.2,
        "kl_target": 0.001,
        "lr": 5e-5,
        "metrics_smoothing_episodes": 5000,
        "model": {
            "custom_model": "cc_model_full_obs",
            "vf_share_layers": False
        },
        "batch_mode": "complete_episodes",
        "num_envs_per_worker": 1,
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_gpus_per_worker": 0.0,
        "num_sgd_iter": 10,
        "rollout_fragment_length": 256,
        "sgd_minibatch_size": 256,
        "train_batch_size": 4096,
        "vf_clip_param": 5.0,
        "vf_share_layers": False,
        "framework": "torch",
        "num_workers": 25,

    }
    return merge_dicts(GRL_DEFAULT_POKER_PPO_PARAMS, config)


def psro_full_bridge_ccppo_params_larger(env: MultiAgentEnv) -> Dict[str, Any]:
    env_config={
        "version": "bridge",
        "fixed_players": True,
    }
    config = {
        "clip_param": 0.03,
        "entropy_coeff": 0.00,
        "framework": "torch",
        "gamma": 1.0,
        "kl_coeff": 0.2,
        "kl_target": 0.001,
        "lr": 5e-4,
        "metrics_smoothing_episodes": 5000,
        "model": {
            "fcnet_hiddens": [256, 128],
            "custom_model": "cc_model_full_obs_larger_bridge",
            "vf_share_layers": False
        },
        "batch_mode": "complete_episodes",
        "num_envs_per_worker": 1,
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_gpus_per_worker": 0.0,
        "num_sgd_iter": 5,
        "rollout_fragment_length": 256,
        "sgd_minibatch_size": 1024,
        "train_batch_size": 4096,
        "vf_clip_param": 10.0,
        "vf_share_layers": False,
        "framework": "torch",
        "num_workers": 25,
    }
    return merge_dicts(GRL_DEFAULT_POKER_PPO_PARAMS, config)

def psro_full_bridge_ccppo_params_indep(env: MultiAgentEnv) -> Dict[str, Any]:
    env_config={
        "version": "bridge",
        "fixed_players": True,
    }
    config = {
        "clip_param": 0.03,
        "entropy_coeff": 0.00,
        "framework": "torch",
        "gamma": 1.0,
        "kl_coeff": 0.2,
        "kl_target": 0.001,
        "lr": 5e-5,
        "metrics_smoothing_episodes": 5000,
        "model": {
            "fcnet_hiddens": [256, 128],
            "custom_model": "cc_model_full_obs_larger_bridge",
            "vf_share_layers": False
        },
        "batch_mode": "complete_episodes",
        "num_envs_per_worker": 1,
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_gpus_per_worker": 0.0,
        "num_sgd_iter": 10,
        "rollout_fragment_length": 256,
        "sgd_minibatch_size": 256,
        "train_batch_size": 4096,
        "vf_clip_param": 5.0,
        "vf_share_layers": False,
        "framework": "torch",
        "num_workers": 5,
    }
    return merge_dicts(GRL_DEFAULT_POKER_PPO_PARAMS, config)
