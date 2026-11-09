"""D2 task registrations using the project PPO and MoE-CTS implementations."""

import gymnasium as gym

if "d2-flat" not in gym.registry:
    gym.register(
        id="d2-flat",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "d2_rl_lab.tasks.d2.env_cfg:D2FlatEnvCfg",
            "play_env_cfg_entry_point": "d2_rl_lab.tasks.d2.env_cfg:D2FlatPlayEnvCfg",
            "rsl_rl_cfg_entry_point": "d2_rl_lab.tasks.d2.rsl_rl_cfg:D2PPORunnerCfg",
        },
    )

if "d2-mixed" not in gym.registry:
    gym.register(
        id="d2-mixed",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "d2_rl_lab.tasks.d2.env_cfg:D2MixedEnvCfg",
            "play_env_cfg_entry_point": "d2_rl_lab.tasks.d2.env_cfg:D2MixedPlayEnvCfg",
            "rsl_rl_cfg_entry_point": "d2_rl_lab.tasks.d2.rsl_rl_cfg:D2PPORunnerCfg",
        },
    )

if "d2-mixed-moe-cts" not in gym.registry:
    gym.register(
        id="d2-mixed-moe-cts",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "d2_rl_lab.tasks.d2.env_cfg:D2MixedEnvCfg",
            "play_env_cfg_entry_point": "d2_rl_lab.tasks.d2.env_cfg:D2MixedPlayEnvCfg",
            "rsl_rl_cfg_entry_point": "d2_rl_lab.tasks.d2.rsl_rl_cfg:D2MoECTSRunnerCfg",
        },
    )
