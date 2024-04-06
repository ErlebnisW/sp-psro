"""An example of customizing PPO to leverage a centralized critic.

Here the model and policy are hard-coded to implement a centralized critic
for TwoStepGame, but you can adapt this for your own use cases.

Compared to simply running `rllib/examples/two_step_game.py --run=PPO`,
this centralized critic version reaches vf_explained_variance=1.0 more stably
since it takes into account the opponent actions as well as the policy's.
Note that this is also using two independent policies instead of weight-sharing
with one.

See also: centralized_critic_2.py for a simpler approach that instead
modifies the environment.
"""

import argparse
import numpy as np
from gym.spaces import Discrete
import os

import ray
from ray import tune
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy, \
    KLCoeffMixin as TorchKLCoeffMixin, ppo_surrogate_loss as torch_loss
from ray.rllib.evaluation.postprocessing import compute_advantages, \
    Postprocessing
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import LearningRateSchedule, \
    EntropyCoeffSchedule
from ray.rllib.policy.torch_policy import LearningRateSchedule as TorchLR, \
    EntropyCoeffSchedule as TorchEntropyCoeffSchedule
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.utils.torch_ops import explained_variance
from ray.rllib.utils.torch_ops import convert_to_torch_tensor
from ray.rllib.utils.torch_ops import apply_grad_clipping
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.policy.torch_policy import TorchPolicy

from ray.rllib.utils import merge_dicts
# from grl.envs.tiny_bridge_2p_multi_agent_env  import TinyBridge2pMultiAgentEnv
from grl.rllib_tools.models.valid_actions_fcnet import get_valid_action_fcn_class_for_env
from grl.rl_apps.centralized_critic_model_full_obs import TorchCentralizedCriticModelFullObs
from grl.rl_apps.centralized_critic_model_full_obs_larger_network import TorchCentralizedCriticModelFullObsLargerModel
from grl.rl_apps.centralized_critic_model_full_obs_larger_network_kuhn import TorchCentralizedCriticModelFullObsLargerModelKuhn
from grl.rl_apps.centralized_critic_model_full_obs_larger_network_tiny_bridge_4p import TorchCentralizedCriticModelFullObsLargerModelTinyBridge
from grl.rl_apps.centralized_critic_model_full_obs_larger_network_bridge_4p import TorchCentralizedCriticModelFullObsLargerModelBridge

from grl.rl_apps.happo import happo_surrogate_loss
torch, nn = try_import_torch()

OPPONENT_OBS_0 = "opponent_obs_0"
OPPONENT_ACTION_0 = "opponent_action_0"
OPPONENT_OBS_1 = "opponent_obs_1"
OPPONENT_ACTION_1 = "opponent_action_1"
PARTNER_OBS = "partner_obs"
PARTNER_ACTION = "partner_action"

parser = argparse.ArgumentParser()
parser.add_argument("--torch", action="store_true")
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--stop-iters", type=int, default=100)
parser.add_argument("--stop-timesteps", type=int, default=100000)
parser.add_argument("--stop-reward", type=float, default=7.99)


class CentralizedValueMixin:
    """Add method to evaluate the central value function from the model."""

    def __init__(self):
        self.compute_central_vf = self.model.central_value_function


# Grabs the opponent obs/act and includes it in the experience train_batch,
# and computes GAE using the central vf predictions.
def centralized_critic_postprocessing_4p_full_obs(policy,
                                      sample_batch,
                                      other_agent_batches=None,
                                      episode=None):
    pytorch = policy.config["framework"] == "torch"

    if (pytorch and hasattr(policy, "compute_central_vf")) or \
            (not pytorch and policy.loss_initialized()):
        assert other_agent_batches is not None
        # other agent batches returns all three other agents, not just one as assumed below
        # TODO: find out which one we need, maybe add in all observations like discussed?
        # print(other_agent_batches, 'other agent batches\n\n')
        # print("#%#$%#$"*20)
        # # print(len(list(other_agent_batches.values())), 'other agent batches len')
        # # print(list(other_agent_batches.keys()), 'other agent keys')
        # print(list(other_agent_batches.values())[0], 'other agent batches 0\n\n')
        # print("#%#$%#$" * 20)
        #
        # print(list(other_agent_batches.values())[1], 'other agent batches 1\n\n')
        # print("#%#$%#$" * 20)

        # print(list(other_agent_batches.values())[2], 'other agent batches 2')

        # TODO: confirm second index is correct
        # TODO: think about if we want to include all player's hidden info
        opp_ids = set(other_agent_batches.keys())
        missing = list(set([0, 1, 2, 3]) - opp_ids)
        assert len(missing) == 1
        my_id = missing[0]
        partner_id = (my_id + 2) % 4
        partner_batch = other_agent_batches[partner_id][1]

        opp_id_0 = (my_id + 1) % 4
        opp_id_1 = (my_id + 3) % 4

        opponent_batch_0 = other_agent_batches[opp_id_0][1]
        opponent_batch_1 = other_agent_batches[opp_id_1][1]

        # [(_, opponent_batch)] = list(other_agent_batches.values())
        # print(opponent_batch, 'opponent batch')

        # also record the opponent obs and actions in the trajectory
        sample_batch[PARTNER_OBS] = partner_batch[SampleBatch.CUR_OBS]
        sample_batch[PARTNER_ACTION] = partner_batch[SampleBatch.ACTIONS]

        sample_batch[OPPONENT_OBS_0] = opponent_batch_0[SampleBatch.CUR_OBS]
        sample_batch[OPPONENT_ACTION_0] = opponent_batch_0[SampleBatch.ACTIONS]

        sample_batch[OPPONENT_OBS_1] = opponent_batch_1[SampleBatch.CUR_OBS]
        sample_batch[OPPONENT_ACTION_1] = opponent_batch_1[SampleBatch.ACTIONS]


        sample_batch[SampleBatch.VF_PREDS] = policy.compute_central_vf(
            convert_to_torch_tensor(sample_batch[SampleBatch.CUR_OBS], policy.device),
            convert_to_torch_tensor(sample_batch[PARTNER_OBS], policy.device),
            convert_to_torch_tensor(sample_batch[PARTNER_ACTION], policy.device),
            convert_to_torch_tensor(sample_batch[OPPONENT_OBS_0], policy.device),
            convert_to_torch_tensor(sample_batch[OPPONENT_ACTION_0], policy.device),
            convert_to_torch_tensor(sample_batch[OPPONENT_OBS_1], policy.device),
            convert_to_torch_tensor(sample_batch[OPPONENT_ACTION_1], policy.device),
        ).cpu().detach().numpy()

    else:
        # Policy hasn't been initialized yet, use zeros.
        sample_batch[PARTNER_OBS] = np.zeros_like(sample_batch[SampleBatch.CUR_OBS])
        sample_batch[PARTNER_ACTION] = np.zeros_like(sample_batch[SampleBatch.ACTIONS])
        sample_batch[OPPONENT_OBS_0] = np.zeros_like(sample_batch[SampleBatch.CUR_OBS])
        sample_batch[OPPONENT_ACTION_0] = np.zeros_like(sample_batch[SampleBatch.ACTIONS])
        sample_batch[OPPONENT_OBS_1] = np.zeros_like(sample_batch[SampleBatch.CUR_OBS])
        sample_batch[OPPONENT_ACTION_1] = np.zeros_like(sample_batch[SampleBatch.ACTIONS])

        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(sample_batch[SampleBatch.REWARDS], dtype=np.float32)

    completed = sample_batch["dones"][-1]
    if completed:
        last_r = 0.0
    else:
        last_r = sample_batch[SampleBatch.VF_PREDS][-1]

    train_batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"])
    return train_batch


def new_func(policy, obs_space, action_space, config):
    CentralizedValueMixin.__init__(policy)


def setup_torch_mixins(policy, obs_space, action_space, config):
    # Copied from PPOTorchPolicy  (w/o ValueNetworkMixin).
    TorchKLCoeffMixin.__init__(policy, config)
    TorchEntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                       config["entropy_coeff_schedule"])
    TorchLR.__init__(policy, config["lr"], config["lr_schedule"])


# def central_vf_stats(policy, train_batch):
#     # Report the explained variance of the central value function.
#     # extra_grad_dict = TorchPolicy.extra_grad_info(policy, train_batch)
#     # new_dict = {
#     #     "central_vf_explained_var": explained_variance(
#     #         train_batch[Postprocessing.VALUE_TARGETS],
#     #         policy._central_value_out),
#     # }
#     old_dict = {
#         "cur_kl_coeff": policy.kl_coeff,
#         "cur_lr": policy.cur_lr,
#         "total_loss": policy._total_loss,
#         "policy_loss": policy._mean_policy_loss,
#         "vf_loss": policy._mean_vf_loss,
#         "vf_explained_var": explained_variance(
#             train_batch[Postprocessing.VALUE_TARGETS],
#             policy.model.value_function()),
#         "kl": policy._mean_kl,
#         "entropy": policy._mean_entropy,
#         "entropy_coeff": policy.entropy_coeff,
#     }
#     # return merge_dicts(new_dict, old_dict)
#     return old_dict

def new_func(policy, obs_space, action_space, config):
    CentralizedValueMixin.__init__(policy)

kuhn_HAPPOTorchPolicy_4P_full_obs_larger = PPOTorchPolicy.with_updates(
    name="kuhnHAPPOTorchPolicy",
    postprocess_fn=centralized_critic_postprocessing_4p_full_obs,
    loss_fn=happo_surrogate_loss,
    before_init=setup_torch_mixins,
    # stats_fn=central_vf_stats,
    after_init=new_func,
    extra_grad_process_fn=apply_grad_clipping,
    mixins=[
        TorchLR, TorchEntropyCoeffSchedule, TorchKLCoeffMixin,
        CentralizedValueMixin
    ])

def get_policy_class(config):
    if config["framework"] == "torch":
        return kuhn_HAPPOTorchPolicy_4P_full_obs_larger


kuhn_HATrainer_4P_full_obs_larger = PPOTrainer.with_updates(
    name="kuhnHAPPOTrainer_full_obs",
    default_policy=None, # changed default to None (used to be TF)
    get_policy_class=get_policy_class,
)

