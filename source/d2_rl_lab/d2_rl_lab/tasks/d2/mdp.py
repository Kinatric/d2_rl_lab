"""D2 contact semantics across physics backends."""

from robot_rl_core import mdp


def base_ground_contact(env, sensor_cfg, threshold):
    """Do not mistake Newton base/thigh self-contact for the body hitting terrain."""
    sensor = env.scene.sensors[sensor_cfg.name]
    matrix = sensor.data.force_matrix_w_history
    if matrix is None:
        if type(sensor).__module__.startswith("isaaclab_newton"):
            raise RuntimeError("D2 Newton base contact requires terrain shape filtering")
        return mdp.illegal_contact(env, sensor_cfg=sensor_cfg, threshold=threshold)
    forces = matrix.torch[:, :, sensor_cfg.body_ids].sum(dim=-2)
    return (forces.norm(dim=-1).max(dim=1).values > threshold).any(dim=-1)
