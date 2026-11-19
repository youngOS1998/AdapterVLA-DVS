"""
Important constants for VLA training and evaluation.

Attempts to automatically identify the correct constants to set based on the Python command used to launch
training or evaluation. If it is unclear, defaults to using the LIBERO simulation benchmark constants.
"""
import sys
from enum import Enum

# Qwen2.5-0.5B token constants
IGNORE_INDEX = -100
# ACTION_TOKEN_BEGIN_IDX = 31743
ACTION_TOKEN_BEGIN_IDX  = 151386
STOP_INDEX = 2  # '</s>'
NUM_TOKENS = 64


# Defines supported normalization schemes for action and proprioceptive state.
class NormalizationType(str, Enum):
    # fmt: off
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]
    # fmt: on


# Define constants for each robot platform
LIBERO_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

CALVIN_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

ALOHA_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 25,
    "ACTION_DIM": 14,
    "PROPRIO_DIM": 14,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}

BRIDGE_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 5,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 7,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

# Single-arm Franka joint-delta (7 Δq + gripper). Used by pick_up / RoboTwin.
FRANKA_JOINT_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 8,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

# Single UR3e: 6 absolute joints + gripper. grasp_conveyor_block is stored as a
# duplicated 14-D vector; the dataset transform keeps the right-arm 7 dims.
UR3E_JOINT_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 12,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 7,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}


# Function to detect robot platform from command line arguments
def _cli_flag_value(flag: str) -> str:
    """Return the value of `--flag value` or `--flag=value`, else empty."""
    args = sys.argv
    prefix = flag + "="
    for i, token in enumerate(args):
        if token == flag and i + 1 < len(args):
            return args[i + 1]
        if token.startswith(prefix):
            return token[len(prefix) :]
    return ""


def detect_robot_platform():
    dataset_name = _cli_flag_value("--dataset_name").lower()
    cmd_args = " ".join(sys.argv).lower()
    # Prefer --dataset_name so sidecar paths cannot flip the platform.
    haystack = dataset_name or cmd_args

    if any(
        key in haystack
        for key in (
            "grasp_conveyor_block",
            "grasp_conveyor_rand_speed_2",
            "grasp_conveyor_rand_speed",
            "grasp_conveyor_multi_objects",
            "grasp_conveyor_dual",
            "grasp_conveyor_ring",
        )
    ):
        return "UR3E_JOINT"
    elif any(key in haystack for key in ("pick_up", "place_color_count_blocks_tray_rand")):
        return "FRANKA_JOINT"
    elif any(key in haystack for key in ("libero",)):
        return "LIBERO"
    elif "aloha" in haystack or "put_back_block" in haystack:
        return "ALOHA"
    elif "bridge" in haystack:
        return "BRIDGE"
    elif "calvin" in haystack:
        return "CALVIN"
    else:
        # Default to LIBERO if unclear
        return "LIBERO"


# Determine which robot platform to use
ROBOT_PLATFORM = detect_robot_platform()

# Set the appropriate constants based on the detected platform
if ROBOT_PLATFORM == "LIBERO":
    constants = LIBERO_CONSTANTS
elif ROBOT_PLATFORM == "FRANKA_JOINT":
    constants = FRANKA_JOINT_CONSTANTS
elif ROBOT_PLATFORM == "UR3E_JOINT":
    constants = UR3E_JOINT_CONSTANTS
elif ROBOT_PLATFORM == "ALOHA":
    constants = ALOHA_CONSTANTS
elif ROBOT_PLATFORM == "BRIDGE":
    constants = BRIDGE_CONSTANTS
elif ROBOT_PLATFORM == "CALVIN":
    constants = CALVIN_CONSTANTS

# Assign constants to global variables
NUM_ACTIONS_CHUNK = constants["NUM_ACTIONS_CHUNK"]
ACTION_DIM = constants["ACTION_DIM"]
PROPRIO_DIM = constants["PROPRIO_DIM"]
ACTION_PROPRIO_NORMALIZATION_TYPE = constants["ACTION_PROPRIO_NORMALIZATION_TYPE"]

# Print which robot platform constants are being used (for debugging)
print(f"Using {ROBOT_PLATFORM} constants:")
print(f"  NUM_ACTIONS_CHUNK = {NUM_ACTIONS_CHUNK}")
print(f"  ACTION_DIM = {ACTION_DIM}")
print(f"  PROPRIO_DIM = {PROPRIO_DIM}")
print(f"  ACTION_PROPRIO_NORMALIZATION_TYPE = {ACTION_PROPRIO_NORMALIZATION_TYPE}")
print("If needed, manually set the correct constants in `prismatic/vla/constants.py`!")
