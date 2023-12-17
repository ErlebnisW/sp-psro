from grl.envs.poker_multi_agent_env import PokerMultiAgentEnv
from grl.rl_apps.scenarios.catalog import scenario_catalog
from grl.rl_apps.scenarios.catalog.common import default_if_creating_ray_head
from grl.rl_apps.scenarios.psro_scenario import PSROScenario
from grl.rl_apps.scenarios.stopping_conditions import *
from grl.rl_apps.scenarios.trainer_configs.poker_psro_configs import *
from grl.rl_apps.scenarios.trainer_configs.bridge_psro_configs import *
from grl.rllib_tools.modified_policies.simple_q_torch_policy import SimpleQTorchPolicyPatched
from ray.rllib.agents.ppo import PPOTrainer, PPOTorchPolicy

# from grl.envs.tiny_bridge_2p_multi_agent_env  import TinyBridge2pMultiAgentEnv
# from grl.rl_apps.tiny_bridge_2p_mappo import CCTrainer, CCPPOTorchPolicy
from grl.rl_apps.tiny_bridge_4p_mappo import CCTrainer_4P, CCPPOTorchPolicy_4P
from grl.rl_apps.tiny_bridge_4p_mappo_full_obs import CCTrainer_4P_full_obs, CCPPOTorchPolicy_4P_full_obs
from grl.rl_apps.tiny_bridge_4p_happo_full_obs import HATrainer_4P_full_obs, HAPPOTorchPolicy_4P_full_obs
from grl.rl_apps.tiny_bridge_4p_mappo_full_obs_larger import CCTrainer_4P_full_obs_larger, CCPPOTorchPolicy_4P_full_obs_larger
from grl.envs.bridge_4p_multi_agent_env import BridgeMultiAgentEnv

scenario_catalog.add(PSROScenario(
    name="tiny_bridge_4p_s_psro",
    ray_cluster_cpus=default_if_creating_ray_head(default=200),
    ray_cluster_gpus=default_if_creating_ray_head(default=1),
    ray_object_store_memory_cap_gigabytes=20,
    env_class=TinyBridge4pMultiAgentEnv,
    env_config={
        "version": "tiny_bridge_4p",
        "fixed_players": True,
    },
    mix_metanash_with_uniform_dist_coeff=0.0,
    allow_stochastic_best_responses=False,
    trainer_class=HATrainer_4P_full_obs,
    policy_classes={
        "metanash": HAPPOTorchPolicy_4P_full_obs,
        "best_response":HAPPOTorchPolicy_4P_full_obs,
        "eval": HAPPOTorchPolicy_4P_full_obs,
    },
    num_eval_workers=8,
    games_per_payoff_eval=1000,
    p2sro=False,
    p2sro_payoff_table_exponential_avg_coeff=None,
    p2sro_sync_with_payoff_table_every_n_episodes=None,
    single_agent_symmetric_game=False,
    get_trainer_config=psro_tiny_bridge_ccppo_params,
    # psro_get_stopping_condition= lambda: StopImmediately(),
    psro_get_stopping_condition=lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
        br_policy_id="best_response",
        dont_check_plateau_before_n_episodes=int(0.8e5),
        check_plateau_every_n_episodes=int(0.8e5),
        minimum_reward_improvement_otherwise_plateaued=0.01,
        max_train_episodes=int(6e6),
    ),
    calc_exploitability_for_openspiel_env=False,
))

scenario_catalog.add(PSROScenario(
    name="tiny_bridge_4p_psro",
    ray_cluster_cpus=default_if_creating_ray_head(default=64),
    ray_cluster_gpus=default_if_creating_ray_head(default=0),
    ray_object_store_memory_cap_gigabytes=20,
    env_class=TinyBridge4pMultiAgentEnv,
    env_config={
        "version": "tiny_bridge_4p",
        "fixed_players": True,
    },
    mix_metanash_with_uniform_dist_coeff=0.0,
    allow_stochastic_best_responses=False,
    trainer_class=CCTrainer_4P_full_obs,
    policy_classes={
        "metanash": CCPPOTorchPolicy_4P_full_obs,
        "best_response": CCPPOTorchPolicy_4P_full_obs,
        "eval": CCPPOTorchPolicy_4P_full_obs,
    },
    num_eval_workers=8,
    games_per_payoff_eval=1000,
    p2sro=False,
    p2sro_payoff_table_exponential_avg_coeff=None,
    p2sro_sync_with_payoff_table_every_n_episodes=None,
    single_agent_symmetric_game=False,
    get_trainer_config=psro_tiny_bridge_ccppo_params,
    # psro_get_stopping_condition= lambda: StopImmediately(),
    psro_get_stopping_condition=lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
        br_policy_id="best_response",
        dont_check_plateau_before_n_episodes=int(0.8e5),
        check_plateau_every_n_episodes=int(0.8e5),
        minimum_reward_improvement_otherwise_plateaued=0.01,
        max_train_episodes=int(6e6),
    ),
    calc_exploitability_for_openspiel_env=False,
))


scenario_catalog.add(PSROScenario(
    name="tiny_bridge_4p_psro_indep",
    ray_cluster_cpus=default_if_creating_ray_head(default=64),
    ray_cluster_gpus=default_if_creating_ray_head(default=0),
    ray_object_store_memory_cap_gigabytes=20,
    env_class=TinyBridge4pMultiAgentEnv,
    env_config={
        "version": "tiny_bridge_4p",
        "fixed_players": True,
    },
    mix_metanash_with_uniform_dist_coeff=0.0,
    allow_stochastic_best_responses=False,
    trainer_class=PPOTrainer,
    policy_classes={
        "metanash": PPOTorchPolicy,
        "best_response": PPOTorchPolicy,
        "eval": PPOTorchPolicy,
    },
    num_eval_workers=8,
    games_per_payoff_eval=1000,
    p2sro=False,
    p2sro_payoff_table_exponential_avg_coeff=None,
    p2sro_sync_with_payoff_table_every_n_episodes=None,
    single_agent_symmetric_game=False,
    get_trainer_config=psro_tiny_bridge_ccppo_params_indep,
    # psro_get_stopping_condition= lambda: StopImmediately(),
    psro_get_stopping_condition=lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
        br_policy_id="best_response",
        dont_check_plateau_before_n_episodes=int(0.8e5),
        check_plateau_every_n_episodes=int(0.8e5),
        minimum_reward_improvement_otherwise_plateaued=0.01,
        max_train_episodes=int(6e6),
    ),
    calc_exploitability_for_openspiel_env=False,
))

scenario_catalog.add(PSROScenario(
    name="tiny_bridge_4p_psro_larger_model",
    ray_cluster_cpus=default_if_creating_ray_head(default=64),
    ray_cluster_gpus=default_if_creating_ray_head(default=0),
    ray_object_store_memory_cap_gigabytes=20,
    env_class=TinyBridge4pMultiAgentEnv,
    env_config={
        "version": "tiny_bridge_4p",
        "fixed_players": True,
    },
    mix_metanash_with_uniform_dist_coeff=0.0,
    allow_stochastic_best_responses=False,
    trainer_class=CCTrainer_4P_full_obs_larger,
    policy_classes={
        "metanash": CCPPOTorchPolicy_4P_full_obs_larger,
        "best_response": CCPPOTorchPolicy_4P_full_obs_larger,
        "eval": CCPPOTorchPolicy_4P_full_obs_larger,
    },
    num_eval_workers=8,
    games_per_payoff_eval=1000,
    p2sro=False,
    p2sro_payoff_table_exponential_avg_coeff=None,
    p2sro_sync_with_payoff_table_every_n_episodes=None,
    single_agent_symmetric_game=False,
    get_trainer_config=psro_tiny_bridge_ccppo_params_larger,
    # psro_get_stopping_condition= lambda: StopImmediately(),
    psro_get_stopping_condition=lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
        br_policy_id="best_response",
        dont_check_plateau_before_n_episodes=int(0.8e5),
        check_plateau_every_n_episodes=int(0.8e5),
        minimum_reward_improvement_otherwise_plateaued=0.01,
        max_train_episodes=int(6e6),
    ),
    calc_exploitability_for_openspiel_env=False,
))

scenario_catalog.add(PSROScenario(
    name="bridge_bidding_psro",
    ray_cluster_cpus=default_if_creating_ray_head(default=64),
    ray_cluster_gpus=default_if_creating_ray_head(default=0),
    ray_object_store_memory_cap_gigabytes=20,
    env_class=BridgeMultiAgentEnv,
    env_config={
        "version": "bridge",
        "fixed_players": True,
        "open_spiel_env_config": {'use_double_dummy_result': True},
    },
    mix_metanash_with_uniform_dist_coeff=0.0,
    allow_stochastic_best_responses=False,
    trainer_class=CCTrainer_4P_full_obs,
    policy_classes={
        "metanash": CCPPOTorchPolicy_4P_full_obs,
        "best_response": CCPPOTorchPolicy_4P_full_obs,
        "eval": CCPPOTorchPolicy_4P_full_obs,
    },
    num_eval_workers=8,
    games_per_payoff_eval=500,
    p2sro=False,
    p2sro_payoff_table_exponential_avg_coeff=None,
    p2sro_sync_with_payoff_table_every_n_episodes=None,
    single_agent_symmetric_game=False,
    get_trainer_config=psro_bridge_ccppo_params,
    # psro_get_stopping_condition= lambda: StopImmediately(),
    psro_get_stopping_condition=lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
        br_policy_id="best_response",
        dont_check_plateau_before_n_episodes=int(4e4),
        check_plateau_every_n_episodes=int(4e4),
        minimum_reward_improvement_otherwise_plateaued=0.01,
        max_train_episodes=int(2.5e5),
    ),
    calc_exploitability_for_openspiel_env=False,
))

scenario_catalog.add(PSROScenario(
    name="bridge_bidding_psro_larger_model",
    ray_cluster_cpus=default_if_creating_ray_head(default=64),
    ray_cluster_gpus=default_if_creating_ray_head(default=0),
    ray_object_store_memory_cap_gigabytes=20,
    env_class=BridgeMultiAgentEnv,
    env_config={
        "version": "bridge",
        "fixed_players": True,
        "open_spiel_env_config": {'use_double_dummy_result': True},
    },
    mix_metanash_with_uniform_dist_coeff=0.0,
    allow_stochastic_best_responses=False,
    trainer_class=CCTrainer_4P_full_obs_larger,
    policy_classes={
        "metanash": CCPPOTorchPolicy_4P_full_obs_larger,
        "best_response": CCPPOTorchPolicy_4P_full_obs_larger,
        "eval": CCPPOTorchPolicy_4P_full_obs_larger,
    },
    num_eval_workers=8,
    games_per_payoff_eval=500,
    p2sro=False,
    p2sro_payoff_table_exponential_avg_coeff=None,
    p2sro_sync_with_payoff_table_every_n_episodes=None,
    single_agent_symmetric_game=False,
    get_trainer_config=psro_bridge_ccppo_params_larger,
    # psro_get_stopping_condition= lambda: StopImmediately(),
    psro_get_stopping_condition=lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
        br_policy_id="best_response",
        dont_check_plateau_before_n_episodes=int(4e4),
        check_plateau_every_n_episodes=int(4e4),
        minimum_reward_improvement_otherwise_plateaued=0.01,
        max_train_episodes=int(2.5e5),
    ),
    calc_exploitability_for_openspiel_env=False,
))

scenario_catalog.add(PSROScenario(
    name="bridge_bidding_psro_indep",
    ray_cluster_cpus=default_if_creating_ray_head(default=64),
    ray_cluster_gpus=default_if_creating_ray_head(default=0),
    ray_object_store_memory_cap_gigabytes=20,
    env_class=BridgeMultiAgentEnv,
    env_config={
        "version": "bridge",
        "fixed_players": True,
        "open_spiel_env_config": {'use_double_dummy_result': True},
    },
    mix_metanash_with_uniform_dist_coeff=0.0,
    allow_stochastic_best_responses=False,
    trainer_class=PPOTrainer,
    policy_classes={
        "metanash": PPOTorchPolicy,
        "best_response": PPOTorchPolicy,
        "eval": PPOTorchPolicy,
    },
    num_eval_workers=5,
    games_per_payoff_eval=500,
    p2sro=False,
    p2sro_payoff_table_exponential_avg_coeff=None,
    p2sro_sync_with_payoff_table_every_n_episodes=None,
    single_agent_symmetric_game=False,
    get_trainer_config=psro_bridge_ccppo_params_indep,
    # psro_get_stopping_condition= lambda: StopImmediately(),
    psro_get_stopping_condition=lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
        br_policy_id="best_response",
        dont_check_plateau_before_n_episodes=int(4e4),
        check_plateau_every_n_episodes=int(4e4),
        minimum_reward_improvement_otherwise_plateaued=0.01,
        max_train_episodes=int(2.5e5),
    ),
    calc_exploitability_for_openspiel_env=False,
))

scenario_catalog.add(PSROScenario(
    name="full_bridge_psro",
    ray_cluster_cpus=default_if_creating_ray_head(default=64),
    ray_cluster_gpus=default_if_creating_ray_head(default=0),
    ray_object_store_memory_cap_gigabytes=20,
    env_class=BridgeMultiAgentEnv,
    env_config={
        "version": "bridge",
        "fixed_players": True,
        "open_spiel_env_config": {'use_double_dummy_result': False},
    },
    mix_metanash_with_uniform_dist_coeff=0.0,
    allow_stochastic_best_responses=False,
    trainer_class=CCTrainer_4P_full_obs,
    policy_classes={
        "metanash": CCPPOTorchPolicy_4P_full_obs,
        "best_response": CCPPOTorchPolicy_4P_full_obs,
        "eval": CCPPOTorchPolicy_4P_full_obs,
    },
    num_eval_workers=8,
    games_per_payoff_eval=500,
    p2sro=False,
    p2sro_payoff_table_exponential_avg_coeff=None,
    p2sro_sync_with_payoff_table_every_n_episodes=None,
    single_agent_symmetric_game=False,
    get_trainer_config=psro_full_bridge_ccppo_params,
    # psro_get_stopping_condition= lambda: StopImmediately(),
    psro_get_stopping_condition=lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
        br_policy_id="best_response",
        dont_check_plateau_before_n_episodes=int(3e4),
        check_plateau_every_n_episodes=int(3e4),
        minimum_reward_improvement_otherwise_plateaued=0.01,
        max_train_episodes=int(1.2e5),
    ),
    calc_exploitability_for_openspiel_env=False,
))


scenario_catalog.add(PSROScenario(
    name="full_bridge_psro_larger_model",
    ray_cluster_cpus=default_if_creating_ray_head(default=64),
    ray_cluster_gpus=default_if_creating_ray_head(default=0),
    ray_object_store_memory_cap_gigabytes=20,
    env_class=BridgeMultiAgentEnv,
    env_config={
        "version": "bridge",
        "fixed_players": True,
        "open_spiel_env_config": {'use_double_dummy_result': False},
    },
    mix_metanash_with_uniform_dist_coeff=0.0,
    allow_stochastic_best_responses=False,
    trainer_class=CCTrainer_4P_full_obs_larger,
    policy_classes={
        "metanash": CCPPOTorchPolicy_4P_full_obs_larger,
        "best_response": CCPPOTorchPolicy_4P_full_obs_larger,
        "eval": CCPPOTorchPolicy_4P_full_obs_larger,
    },
    num_eval_workers=8,
    games_per_payoff_eval=500,
    p2sro=False,
    p2sro_payoff_table_exponential_avg_coeff=None,
    p2sro_sync_with_payoff_table_every_n_episodes=None,
    single_agent_symmetric_game=False,
    get_trainer_config=psro_full_bridge_ccppo_params_larger,
    # psro_get_stopping_condition= lambda: StopImmediately(),
    psro_get_stopping_condition=lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
        br_policy_id="best_response",
        dont_check_plateau_before_n_episodes=int(3e4),
        check_plateau_every_n_episodes=int(3e4),
        minimum_reward_improvement_otherwise_plateaued=0.01,
        max_train_episodes=int(1.2e5),
    ),
    calc_exploitability_for_openspiel_env=False,
))

scenario_catalog.add(PSROScenario(
    name="full_bridge_psro_indep",
    ray_cluster_cpus=default_if_creating_ray_head(default=64),
    ray_cluster_gpus=default_if_creating_ray_head(default=0),
    ray_object_store_memory_cap_gigabytes=20,
    env_class=BridgeMultiAgentEnv,
    env_config={
        "version": "bridge",
        "fixed_players": True,
        "open_spiel_env_config": {'use_double_dummy_result': False},
    },
    mix_metanash_with_uniform_dist_coeff=0.0,
    allow_stochastic_best_responses=False,
    trainer_class=PPOTrainer,
    policy_classes={
        "metanash": PPOTorchPolicy,
        "best_response": PPOTorchPolicy,
        "eval": PPOTorchPolicy,
    },
    num_eval_workers=5,
    games_per_payoff_eval=500,
    p2sro=False,
    p2sro_payoff_table_exponential_avg_coeff=None,
    p2sro_sync_with_payoff_table_every_n_episodes=None,
    single_agent_symmetric_game=False,
    get_trainer_config=psro_full_bridge_ccppo_params_indep,
    # psro_get_stopping_condition= lambda: StopImmediately(),
    psro_get_stopping_condition=lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
        br_policy_id="best_response",
        dont_check_plateau_before_n_episodes=int(3e4),
        check_plateau_every_n_episodes=int(3e4),
        minimum_reward_improvement_otherwise_plateaued=0.01,
        max_train_episodes=int(1.2e5),
    ),
    calc_exploitability_for_openspiel_env=False,
))
