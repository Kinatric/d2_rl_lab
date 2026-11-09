"""Select the robot adapter from the existing YAML robot field."""


def make_robot_adapter(config, policy_path, explicit_xml=None):
    robot = config.get("robot")
    if robot == "d2":
        from d2_rl_lab.deployment.mujoco import D2MujocoAdapter
        return D2MujocoAdapter(config, policy_path, explicit_xml)
    raise ValueError(f"Unsupported robot {robot!r}; this checkout supports 'd2'")
