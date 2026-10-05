"""Berkeley Humanoid joystick task, straight from MuJoCo Playground.
"""

from mujoco_playground._src.locomotion.berkeley_humanoid import joystick as _joystick

BerkeleyHumanoidJoystick = _joystick.Joystick
default_config = _joystick.default_config

JOINT_NAMES = (
    "LL_HR", "LL_HAA", "LL_HFE", "LL_KFE", "LL_FFE", "LL_FAA",
    "LR_HR", "LR_HAA", "LR_HFE", "LR_KFE", "LR_FFE", "LR_FAA",
)

JOINT_ACTUATOR_FORCE_LIMIT = (20.0, 20.0, 30.0, 30.0, 20.0, 5.0) * 2
PROVISIONAL_TORQUE_LIMIT = tuple(0.1 * x for x in JOINT_ACTUATOR_FORCE_LIMIT)

__all__ = [
    "BerkeleyHumanoidJoystick",
    "default_config",
    "JOINT_NAMES",
    "JOINT_ACTUATOR_FORCE_LIMIT",
    "PROVISIONAL_TORQUE_LIMIT",
]
