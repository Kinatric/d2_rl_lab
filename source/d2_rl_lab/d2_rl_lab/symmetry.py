"""D2 joint/foot symmetry; preserves the trained observation/action mapping."""
from robot_rl_core.utils.symmetry import Symmetry, SymmetryMapper


class D2SymmetryMapper(SymmetryMapper):
    JOINT_PERM = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
    JOINT_SIGN = [-1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1]
    FOOT_PERM = [1, 0, 3, 2]


class D2Symmetry(Symmetry):
    mapper_type = D2SymmetryMapper
