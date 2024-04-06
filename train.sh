export CUDA_VISIBLE_DEVICES=0
# python grl/rl_apps/psro/team_self_play_4p.py
# python grl/rl_apps/psro/team_psro_manager.py --scenario tiny_bridge_4p_s_psro
# python grl/rl_apps/psro/team_psro_manager.py --scenario kuhn_4p_s_psro
# python grl/rl_apps/psro/team_psro_manager.py --scenario tiny_bridge_4p_psro

# python grl/rl_apps/psro/general_psro_eval.py --scenario leduc_psro_ppo_discrete_action


python grl/rl_apps/psro/general_psro_manager.py --scenario leduc_psro_ppo_discrete_action