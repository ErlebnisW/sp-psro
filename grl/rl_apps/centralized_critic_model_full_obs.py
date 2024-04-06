### directly adapts from rllib's centralized_critique_models.py

from gym.spaces import Box
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models import ModelCatalog

from grl.rllib_tools.models.valid_actions_fcnet import get_valid_action_fcn_class

torch, nn = try_import_torch()

from torch.optim import Adam

class TorchCentralizedCriticModelFullObs(TorchModelV2, nn.Module):
    """Multi-agent model that implements a centralized VF."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)  ### TODO: check this model_config
        nn.Module.__init__(self)
        self.other_policies = {}

        # tiny bridge 2p
        # self.obs_dim = 22
        # self.action_dim = 7

        # tiny bridge 4p
        self.obs_dim = 84
        self.action_dim = 9

        # kuhn 4p
        # self.obs_dim = 23
        # self.action_dim = 2
        # input_size = (self.obs_dim + self.action_dim)*4 + self.action_dim*3

        # goofspiel 4p
        # self.obs_dim = 61
        # self.action_dim = 3

        # bridge
        # self.obs_dim = 571
        # self.action_dim = 90

        # Base of the model
        self.model = get_valid_action_fcn_class(self.obs_dim, self.action_dim)(obs_space, action_space, num_outputs,
                                                       model_config,
                                                       name)
        ### can figure this out by renaming this and never instantiate self. model
        # Central VF maps (obs, opp_obs, opp_act) -> vf_pred
        input_size = 399  # obs + opp_obs + opp_act
        self.central_vf = nn.Sequential(
            SlimFC(input_size, 16, activation_fn=nn.Tanh),
            SlimFC(16, 1),
        )
        # self.central_vf = nn.Sequential(
        #     SlimFC(input_size, 64, activation_fn=nn.ReLU),
        #     SlimFC(64, 32, activation_fn=nn.ReLU),
        #     SlimFC(32, 16, activation_fn=nn.ReLU),
        #     SlimFC(16, 1),
        # )
        self.custom_config = {
            "clip_param": 0.03,
            "entropy_coeff": 0.00,
            "framework": "torch",
            "gamma": 1.0,
            "kl_coeff": 0.2,
            "kl_target": 0.001,
            "critic_lr": 5e-5,
            "actor_lr":5e-5,
            "metrics_smoothing_episodes": 5000,
            "model": {
                "custom_model": "cc_model_full_obs",
                "vf_share_layers": False
            },
            "batch_mode": "complete_episodes",
            "num_envs_per_worker": 1,
            # "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
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
        
        self.actor_optimizer = Adam(params=self.parameters(), lr=self.custom_config["actor_lr"])

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        model_out, _ = self.model(input_dict, state, seq_lens)
        # raise ValueError(model_out, type(self.model), input_dict, state, seq_lens)
        return model_out, []

    def central_value_function(self, obs, partner_obs, partner_actions, opponent_obs_0,
                               opponent_actions_0, opponent_obs_1, opponent_actions_1):
        if obs.shape[0] == opponent_obs_0.shape[0]:
            # print(1/0)
            input_ = torch.cat([
                obs,
                partner_obs,
                torch.nn.functional.one_hot(partner_actions, self.action_dim).float(),
                opponent_obs_0,
                torch.nn.functional.one_hot(opponent_actions_0, self.action_dim).float(),
                opponent_obs_1,
                torch.nn.functional.one_hot(opponent_actions_1, self.action_dim).float(),
            ], 1)
            # print("### C success!", obs.shape,  opponent_obs.shape) ## return has the same shape[0] as input
            return torch.reshape(self.central_vf(input_), [-1])
        else:
            import pdb;
            pdb.set_trace()
            print(1 / 0)

    @override(ModelV2)
    def value_function(self):
        return self.model.value_function()  # not used
    
    
    def update_actor(self, loss, lr, grad_clip):
        TorchCentralizedCriticModelFullObs.update_use_torch_adam(
            loss=(-1 * loss),
            optimizer=self.actor_optimizer,
            parameters=self.parameters(),
            grad_clip=grad_clip
        )

    @staticmethod
    def update_use_torch_adam(loss, parameters, optimizer, grad_clip):
        optimizer.zero_grad()
        loss.backward()
        # total_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in parameters if p.grad is not None]))
        if grad_clip is not None: 
            torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
        optimizer.step()
    
ModelCatalog.register_custom_model("cc_model_full_obs", TorchCentralizedCriticModelFullObs)