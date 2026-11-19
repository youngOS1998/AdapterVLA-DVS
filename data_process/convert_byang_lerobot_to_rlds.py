#!/usr/bin/env python3
"""Convert LeRobot v2.1 or RoboTwin hdf5 demos to single-arm Franka RLDS.

Supported sources
-----------------
LeRobot v2.1 (byang Franka)::

    data/byang_datasets/pick_up/
        meta/info.json
        data/chunk-000/episode_000000.parquet
        videos/chunk-000/observation.images.{base,wrist}/episode_000000.mp4

RoboTwin / XPolicyLab hdf5 (single right-arm Franka)::

    /mnt/Projects/RoboTwin/data/demo_clean_franka/place_color_count_blocks_tray_rand
        franka_panda_franka_panda/data/episode_0000000.hdf5
        franka_panda_franka_panda/instruction/episode_0000000.json

RoboTwin Franka notes: the left arm is parked, so only the right arm is exported.
Images are ``cam_head`` -> ``image`` and ``cam_right_wrist`` -> ``wrist_image``.

RoboTwin Aloha AgileX (bimanual, 6-DoF arms), e.g. ``data/put_back_block``::

    data/put_back_block/aloha_agilex/data/episode_0000000.hdf5
    data/put_back_block/aloha_agilex/instruction/episode_0000000.json

Aloha steps are 14-D absolute joints, not Franka joint deltas::

    action                         (14,)  left q6 + grip + right q6 + grip
    observation.image              (256, 256, 3)  cam_head
    observation.left_wrist_image   (256, 256, 3)  cam_left_wrist
    observation.right_wrist_image  (256, 256, 3)  cam_right_wrist
    observation.state              (14,)  same layout as action, from ``state/``
    language_instruction           task text

Train Aloha with ALOHA constants (ACTION_DIM=14, PROPRIO_DIM=14, 3 cameras)
and ``aloha_dataset_transform``. Embodiment is detected from the hdf5 joint width.

Each step (joint-delta action space)::

    action                  (8,)   Δq1..Δq7 (rad) + gripper
    observation.image       (256, 256, 3)  base / head RGB
    observation.wrist_image (256, 256, 3)  wrist RGB
    observation.state       (8,)   q1..q7 + gripper
    observation.joint_state (7,)   Franka joint angles
    language_instruction    task text

Train in this repo with FRANKA_JOINT constants (ACTION_DIM=8, PROPRIO_DIM=8)
and ``franka_joint_dataset_transform``.

Examples::

    /root/anaconda3/envs/vla-adapter/bin/python \\
        data_process/convert_byang_lerobot_to_rlds.py --overwrite

    /root/anaconda3/envs/vla-adapter/bin/python \\
        data_process/convert_byang_lerobot_to_rlds.py --max_episodes 2 --overwrite

    /root/anaconda3/envs/vla-adapter/bin/python \\
        data_process/convert_byang_lerobot_to_rlds.py \\
        --source_dir /mnt/Projects_origin/VLA-Adapter/data/put_back_block \\
        --dataset_name put_back_block --overwrite

    cd /mnt/Projects_origin/VLA-Adapter

    /root/anaconda3/envs/vla-adapter/bin/python \
    data_process/convert_byang_lerobot_to_rlds.py \
    --source_dir /mnt/Projects_origin/VLA-Adapter/data/grasp_conveyor_rand_speed \
    --output_dir /mnt/Projects_origin/VLA-Adapter/data/robotwin_rlds \
    --dataset_name grasp_conveyor_rand_speed

    /root/anaconda3/envs/vla-adapter/bin/python \
    data_process/convert_byang_lerobot_to_rlds.py \
    --source_dir /mnt/Projects_origin/VLA-Adapter/data/grasp_conveyor_rand_speed_2 \
    --output_dir /mnt/Projects_origin/VLA-Adapter/data/robotwin_rlds \
    --dataset_name grasp_conveyor_rand_speed_2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from io import BytesIO
from pathlib import Path
from typing import Optional

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("NO_GCE_CHECK", "true")

import numpy as np
from PIL import Image

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core.utils import gcs_utils

gcs_utils._is_gcs_disabled = True

DEFAULT_SOURCE = Path("/mnt/Projects/RoboTwin/data/demo_clean_franka")
DEFAULT_OUTPUT = Path("/mnt/Projects_origin/VLA-Adapter/data/robotwin_rlds")
IMAGE_HW = (256, 256)  # H, W
ACTION_DIM = 8  # 7 joint deltas + gripper
STATE_DIM = 8  # 7 joints + gripper
JOINT_DIM = 7
# Aloha AgileX: 2 x (6 joints + gripper), absolute joint targets.
ALOHA_ACTION_DIM = 14
ALOHA_STATE_DIM = 14
ALOHA_JOINT_DIM = 6
NOOP_THRESH = 1e-3
EPISODE_RE = re.compile(r"episode[_\-]?(\d+)", re.IGNORECASE)
DEFAULT_ARM = "right"

# Franka Panda modified DH (official control parameters), flange included.
# Each row: (a, d, alpha, theta_offset). Used only for LeRobot joint-only dumps.
_PANDA_MDH = (
    (0.0, 0.333, 0.0, 0.0),
    (0.0, 0.0, -math.pi / 2.0, 0.0),
    (0.0, 0.316, math.pi / 2.0, 0.0),
    (0.0825, 0.0, math.pi / 2.0, 0.0),
    (-0.0825, 0.384, -math.pi / 2.0, 0.0),
    (0.0, 0.0, math.pi / 2.0, 0.0),
    (0.088, 0.0, math.pi / 2.0, 0.0),
    (0.0, 0.107, 0.0, 0.0),  # flange, theta = 0
)


def _require_pyarrow():
    try:
        import pyarrow.parquet as pq  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required to read LeRobot parquet files. Install with:\n"
            "  /root/anaconda3/envs/vla-adapter/bin/pip install pyarrow"
        ) from exc
    return pq


def _mdh_transform(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    ca, sa = math.cos(alpha), math.sin(alpha)
    ct, st = math.cos(theta), math.sin(theta)
    return np.array(
        [
            [ct, -st, 0.0, a],
            [st * ca, ct * ca, -sa, -d * sa],
            [st * sa, ct * sa, ca, d * ca],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def panda_fk(joints7: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (position xyz, rotation 3x3) of the Panda flange."""
    q = np.asarray(joints7, dtype=np.float64).reshape(-1)
    if q.size < 7:
        raise ValueError(f"need 7 joints, got {q.size}")
    t = np.eye(4, dtype=np.float64)
    for i, (a, d, alpha, theta_off) in enumerate(_PANDA_MDH):
        theta = (q[i] + theta_off) if i < 7 else theta_off
        t = t @ _mdh_transform(a, d, alpha, theta)
    return t[:3, 3].copy(), t[:3, :3].copy()


def rot_to_rpy(rot: np.ndarray) -> np.ndarray:
    """XYZ Euler (roll, pitch, yaw), same convention as LIBERO EEF state."""
    r20 = float(np.clip(rot[2, 0], -1.0, 1.0))
    pitch = math.asin(-r20)
    if abs(r20) < 0.999999:
        roll = math.atan2(rot[2, 1], rot[2, 2])
        yaw = math.atan2(rot[1, 0], rot[0, 0])
    else:
        roll = math.atan2(-rot[0, 1], rot[1, 1])
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float32)


def rpy_delta(rot_src: np.ndarray, rot_dst: np.ndarray) -> np.ndarray:
    return rot_to_rpy(rot_src.T @ rot_dst)


def quat_wxyz_to_rot(quat: np.ndarray) -> np.ndarray:
    """transforms3d / RoboTwin quaternion (w, x, y, z) -> 3x3 rotation."""
    q = np.asarray(quat, dtype=np.float64).reshape(-1)[:4]
    n = math.sqrt(float(np.dot(q, q))) or 1.0
    w, x, y, z = (q / n).tolist()
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose7_to_pos_rot(pose7: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pose7, dtype=np.float64).reshape(-1)
    return p[:3].copy(), quat_wxyz_to_rot(p[3:7])


def joints_to_proprio(joints: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    """(T, 8) = current joint angles + gripper."""
    q = np.asarray(joints, dtype=np.float32).reshape(-1, JOINT_DIM)
    g = np.clip(np.asarray(gripper, dtype=np.float32).reshape(-1, 1), 0.0, 1.0)
    return np.concatenate([q, g], axis=-1)


def joint_delta_actions(
    state_joints: np.ndarray,
    action_joints: np.ndarray,
    action_grip: np.ndarray,
) -> np.ndarray:
    """8-D action: commanded joints minus current joints, plus gripper."""
    q_s = np.asarray(state_joints, dtype=np.float32).reshape(-1, JOINT_DIM)
    q_a = np.asarray(action_joints, dtype=np.float32).reshape(-1, JOINT_DIM)
    g = np.clip(np.asarray(action_grip, dtype=np.float32).reshape(-1, 1), 0.0, 1.0)
    return np.concatenate([q_a - q_s, g], axis=-1)


def _clip_grip(gripper: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(gripper, dtype=np.float32).reshape(-1, 1), 0.0, 1.0)


def pack_aloha_vector(
    left_q: np.ndarray,
    left_g: np.ndarray,
    right_q: np.ndarray,
    right_g: np.ndarray,
) -> np.ndarray:
    """14-D Aloha vector: left q6 + grip + right q6 + grip."""
    return np.concatenate(
        [
            np.asarray(left_q, dtype=np.float32).reshape(-1, ALOHA_JOINT_DIM),
            _clip_grip(left_g),
            np.asarray(right_q, dtype=np.float32).reshape(-1, ALOHA_JOINT_DIM),
            _clip_grip(right_g),
        ],
        axis=-1,
    )


def joints_to_eef_state(joints: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    """(T, 8) = xyz + rpy + [grip, grip]. Kept for reference / debugging."""
    t_len = int(joints.shape[0])
    out = np.zeros((t_len, 8), dtype=np.float32)
    grip = np.asarray(gripper, dtype=np.float32).reshape(-1)
    for i in range(t_len):
        pos, rot = panda_fk(joints[i])
        rpy = rot_to_rpy(rot)
        g = float(np.clip(grip[i], 0.0, 1.0))
        out[i] = np.array([*pos, *rpy, g, g], dtype=np.float32)
    return out


def eef_delta_actions(
    state_joints: np.ndarray,
    action_joints: np.ndarray,
    action_grip: np.ndarray,
) -> np.ndarray:
    """7-D EEF delta (legacy). Not used by the joint-delta converter."""
    t_len = int(state_joints.shape[0])
    out = np.zeros((t_len, 7), dtype=np.float32)
    grip = np.asarray(action_grip, dtype=np.float32).reshape(-1)
    for i in range(t_len):
        pos_s, rot_s = panda_fk(state_joints[i])
        pos_a, rot_a = panda_fk(action_joints[i])
        dpos = pos_a - pos_s
        drpy = rpy_delta(rot_s, rot_a)
        g = float(np.clip(grip[i], 0.0, 1.0))
        out[i] = np.array([*dpos, *drpy, g], dtype=np.float32)
    return out


def poses_to_eef_state(poses: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    """RoboTwin right_ee_poses (xyz + wxyz) -> LIBERO 8-D EEF state."""
    t_len = int(poses.shape[0])
    out = np.zeros((t_len, STATE_DIM), dtype=np.float32)
    grip = np.asarray(gripper, dtype=np.float32).reshape(-1)
    for i in range(t_len):
        pos, rot = pose7_to_pos_rot(poses[i])
        rpy = rot_to_rpy(rot)
        g = float(np.clip(grip[i], 0.0, 1.0))
        out[i] = np.array([*pos, *rpy, g, g], dtype=np.float32)
    return out


def poses_to_eef_delta(
    state_poses: np.ndarray,
    action_poses: np.ndarray,
    action_grip: np.ndarray,
) -> np.ndarray:
    """7-D LIBERO action from RoboTwin current vs commanded EE poses."""
    t_len = int(state_poses.shape[0])
    out = np.zeros((t_len, ACTION_DIM), dtype=np.float32)
    grip = np.asarray(action_grip, dtype=np.float32).reshape(-1)
    for i in range(t_len):
        pos_s, rot_s = pose7_to_pos_rot(state_poses[i])
        pos_a, rot_a = pose7_to_pos_rot(action_poses[i])
        dpos = pos_a - pos_s
        drpy = rpy_delta(rot_s, rot_a)
        g = float(np.clip(grip[i], 0.0, 1.0))
        out[i] = np.array([*dpos, *drpy, g], dtype=np.float32)
    return out


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_tasks(meta_dir: Path) -> dict[int, str]:
    tasks = {}
    for row in load_jsonl(meta_dir / "tasks.jsonl"):
        tasks[int(row["task_index"])] = str(row["task"])
    return tasks


def load_episodes(meta_dir: Path) -> list[dict]:
    return load_jsonl(meta_dir / "episodes.jsonl")


def read_parquet_episode(path: Path) -> dict[str, np.ndarray]:
    pq = _require_pyarrow()
    table = pq.read_table(path)
    names = set(table.column_names)

    def col(name: str):
        if name not in names:
            raise KeyError(f"{path} missing column {name}")
        return table.column(name).to_pylist()

    state = np.asarray(col("observation.state"), dtype=np.float32)
    action = np.asarray(col("action"), dtype=np.float32)
    if state.ndim != 2 or state.shape[1] < 8:
        raise ValueError(f"{path} observation.state shape {state.shape}, expected (T, 8)")
    if action.ndim != 2 or action.shape[1] < 8:
        raise ValueError(f"{path} action shape {action.shape}, expected (T, 8)")

    task_index = np.asarray(col("task_index"), dtype=np.int64).reshape(-1)
    done = np.asarray(col("next.done"), dtype=np.bool_) if "next.done" in names else None
    return {
        "state": state,
        "action": action,
        "task_index": task_index,
        "done": done,
    }


def _resize_rgb(frame: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    img = Image.fromarray(np.asarray(frame)).convert("RGB")
    if img.size != (hw[1], hw[0]):
        img = img.resize((hw[1], hw[0]), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _decode_jpeg(raw, hw: tuple[int, int]) -> np.ndarray:
    payload = raw.tobytes() if isinstance(raw, np.ndarray) else bytes(raw)
    img = Image.open(BytesIO(payload)).convert("RGB")
    if img.size != (hw[1], hw[0]):
        img = img.resize((hw[1], hw[0]), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def decode_video(path: Path, hw: tuple[int, int]) -> np.ndarray:
    try:
        import cv2
    except ImportError:
        cv2 = None

    frames: list[np.ndarray] = []
    if cv2 is not None:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video {path}")
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                frames.append(_resize_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), hw))
        finally:
            cap.release()
    else:
        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise ImportError("need opencv-python or imageio to decode mp4") from exc
        reader = imageio.get_reader(str(path))
        try:
            for frame in reader:
                frames.append(_resize_rgb(frame, hw))
        finally:
            reader.close()

    if not frames:
        raise RuntimeError(f"no frames in {path}")
    return np.stack(frames, axis=0)


def placeholder_video(n: int, hw: tuple[int, int]) -> np.ndarray:
    return np.zeros((n, hw[0], hw[1], 3), dtype=np.uint8)


def episode_chunk(episode_index: int, chunks_size: int) -> int:
    return int(episode_index) // int(chunks_size)


def format_lerobot_path(template: str, episode_index: int, chunks_size: int, video_key: str = "") -> str:
    return template.format(
        episode_chunk=episode_chunk(episode_index, chunks_size),
        episode_index=int(episode_index),
        video_key=video_key,
    )


def _is_lerobot_root(path: Path) -> bool:
    return (path / "meta" / "info.json").is_file()


def _is_robotwin_env_dir(path: Path) -> bool:
    data_dir = path / "data"
    return data_dir.is_dir() and any(data_dir.glob("episode*.hdf5"))


def detect_robotwin_embodiment(source: Path) -> str:
    """Return ``aloha`` for 6-DoF bimanual arms, otherwise ``franka``."""
    import h5py

    data_dir = source / "data"
    hdf5_files = sorted(data_dir.glob("episode*.hdf5"), key=_episode_id_from_path)
    if not hdf5_files:
        return "franka"
    with h5py.File(hdf5_files[0], "r") as h5:
        left_key = "state/left_arm_joint_states"
        right_key = "state/right_arm_joint_states"
        if left_key in h5 and right_key in h5:
            left_n = int(h5[left_key].shape[-1])
            right_n = int(h5[right_key].shape[-1])
            if left_n == ALOHA_JOINT_DIM and right_n == ALOHA_JOINT_DIM:
                return "aloha"
    return "franka"


def _episode_id_from_path(path: Path) -> int:
    match = EPISODE_RE.search(path.stem)
    if not match:
        raise ValueError(f"cannot parse episode id from {path.name}")
    return int(match.group(1))


def find_instruction_dir(source: Path) -> Path:
    for name in ("instruction", "instructions"):
        candidate = source / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"missing instruction(s) directory under {source}")


def find_instruction_file(inst_dir: Path, episode_id: int) -> Path:
    padded = f"{episode_id:07d}"
    for name in (f"episode_{padded}.json", f"episode{episode_id}.json", f"episode_{episode_id}.json"):
        path = inst_dir / name
        if path.is_file():
            return path
    return inst_dir / f"episode_{padded}.json"


def _as_str(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def robotwin_instruction(inst_path: Path, episode_id: int, hdf5_fallback: str) -> str:
    candidates: list[str] = []
    if inst_path.is_file():
        with inst_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            candidates = [str(x) for x in (payload.get("seen") or payload.get("unseen") or [])]
        elif isinstance(payload, list):
            candidates = [str(x) for x in payload]
    if not candidates and hdf5_fallback:
        try:
            parsed = json.loads(hdf5_fallback)
            if isinstance(parsed, dict):
                candidates = [str(x) for x in (parsed.get("seen") or parsed.get("unseen") or [])]
            elif isinstance(parsed, list):
                candidates = [str(x) for x in parsed]
        except json.JSONDecodeError:
            candidates = [hdf5_fallback]
    texts = [t.strip() for t in candidates if str(t).strip()]
    if not texts:
        return "complete the task."
    rng = np.random.RandomState(episode_id)
    return texts[int(rng.randint(len(texts)))]


def read_robotwin_arm(h5, arm: str) -> dict[str, np.ndarray]:
    """Load one arm's joints / gripper / EE poses. ``arm`` is ``right`` or ``left``."""
    prefix_state = "state"
    prefix_action = "action"
    joints = np.asarray(h5[f"{prefix_state}/{arm}_arm_joint_states"][:], dtype=np.float32)
    act_joints = np.asarray(h5[f"{prefix_action}/{arm}_arm_joint_states"][:], dtype=np.float32)
    state_grip = np.asarray(h5[f"{prefix_state}/{arm}_ee_joint_states"][:], dtype=np.float32).reshape(-1)
    act_grip = np.asarray(h5[f"{prefix_action}/{arm}_ee_joint_states"][:], dtype=np.float32).reshape(-1)
    state_pose = np.asarray(h5[f"{prefix_state}/{arm}_ee_poses"][:], dtype=np.float32)
    act_pose = np.asarray(h5[f"{prefix_action}/{arm}_ee_poses"][:], dtype=np.float32)
    if joints.ndim != 2 or joints.shape[1] < 7:
        raise ValueError(f"{arm} joints shape {joints.shape}, expected (T, 7)")
    if state_pose.ndim != 2 or state_pose.shape[1] < 7:
        raise ValueError(f"{arm} ee_poses shape {state_pose.shape}, expected (T, 7) xyz+wxyz")
    return {
        "joints": joints[:, :7],
        "act_joints": act_joints[:, :7],
        "state_grip": state_grip,
        "act_grip": act_grip,
        "state_pose": state_pose[:, :7],
        "act_pose": act_pose[:, :7],
    }


def discover_dataset_roots(source_dir: Path) -> list[tuple[str, Path, str]]:
    """Return [(kind, root, default_dataset_name), ...]. kind is lerobot|robotwin."""
    source_dir = source_dir.expanduser().resolve()
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)

    if _is_lerobot_root(source_dir):
        return [("lerobot", source_dir, source_dir.name)]
    if _is_robotwin_env_dir(source_dir):
        name = source_dir.parent.name if source_dir.parent.name else source_dir.name
        return [("robotwin", source_dir, name)]

    found: list[tuple[str, Path, str]] = []
    direct_env: list[Path] = []
    for child in sorted(p for p in source_dir.iterdir() if p.is_dir()):
        if _is_lerobot_root(child):
            found.append(("lerobot", child, child.name))
            continue
        if _is_robotwin_env_dir(child):
            direct_env.append(child)
            continue
        env_kids = [p for p in child.iterdir() if p.is_dir() and _is_robotwin_env_dir(p)]
        if len(env_kids) == 1:
            found.append(("robotwin", env_kids[0], child.name))
        elif env_kids:
            preferred = [p for p in env_kids if "franka" in p.name.lower()]
            pick = preferred[0] if preferred else env_kids[0]
            found.append(("robotwin", pick, child.name))

    if len(direct_env) == 1:
        # put_back_block/aloha_agilex -> dataset name put_back_block
        found.append(("robotwin", direct_env[0], source_dir.name))
    else:
        for child in direct_env:
            found.append(("robotwin", child, child.name))

    if not found:
        raise FileNotFoundError(
            f"no LeRobot (meta/info.json) or RoboTwin (data/episode*.hdf5) dataset under {source_dir}"
        )
    return found


def count_videos(root: Path, info: dict, key: str) -> int:
    video_tmpl = info["video_path"]
    chunks_size = int(info.get("chunks_size", 1000))
    n = 0
    for ep in range(int(info["total_episodes"])):
        rel = format_lerobot_path(video_tmpl, ep, chunks_size, key)
        if (root / rel).is_file():
            n += 1
    return n


def _pack_steps(
    *,
    n: int,
    base: np.ndarray,
    wrist: np.ndarray,
    proprio: np.ndarray,
    action: np.ndarray,
    joints: np.ndarray,
    instruction: str,
    skip_noops: bool,
) -> list[dict]:
    keep = np.ones(n, dtype=bool)
    if skip_noops:
        mag = np.linalg.norm(action[:, :JOINT_DIM], axis=1)
        keep = mag >= NOOP_THRESH
        keep[0] = True
        keep[-1] = True
        if int(keep.sum()) < 2:
            keep[:] = True

    steps = []
    indices = np.flatnonzero(keep)
    last_k = int(indices[-1])
    for step_i, t in enumerate(indices):
        is_last = t == last_k
        steps.append(
            {
                "observation": {
                    "image": base[t],
                    "wrist_image": wrist[t],
                    "state": proprio[t],
                    "joint_state": joints[t].astype(np.float32),
                },
                "action": action[t],
                "discount": np.float32(1.0),
                "reward": np.float32(1.0 if is_last else 0.0),
                "is_first": step_i == 0,
                "is_last": is_last,
                "is_terminal": is_last,
                "language_instruction": instruction,
            }
        )
    return steps


class ByangPickUp(tfds.core.GeneratorBasedBuilder):
    """LeRobot / RoboTwin Franka demos converted to 8-D joint-delta RLDS."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Converted to 8-D joint-delta RLDS (Δq7 + gripper)."}
    SOURCE_DIR = ""
    SOURCE_KIND = "lerobot"
    MAX_EPISODES = None
    SKIP_NOOPS = False
    PLACEHOLDER_IMAGES = False
    INVERT_GRIPPER = False
    IMAGE_HW = IMAGE_HW
    ARM = DEFAULT_ARM
    EMBODIMENT = "franka"  # franka | aloha
    DATASET_DESCRIPTION = "Franka demos converted to 8-D joint-delta RLDS."

    def _info(self) -> tfds.core.DatasetInfo:
        h, w = self.IMAGE_HW

        def rgb():
            return tfds.features.Image(shape=(h, w, 3), dtype=np.uint8, encoding_format="jpeg")

        if self.EMBODIMENT == "aloha":
            observation = {
                "image": rgb(),
                "left_wrist_image": rgb(),
                "right_wrist_image": rgb(),
                "state": tfds.features.Tensor(shape=(ALOHA_STATE_DIM,), dtype=np.float32),
            }
            action_dim = ALOHA_ACTION_DIM
        else:
            observation = {
                "image": rgb(),
                "wrist_image": rgb(),
                "state": tfds.features.Tensor(shape=(STATE_DIM,), dtype=np.float32),
                "joint_state": tfds.features.Tensor(shape=(JOINT_DIM,), dtype=np.float32),
            }
            action_dim = ACTION_DIM
        return tfds.core.DatasetInfo(
            builder=self,
            description=self.DATASET_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(observation),
                            "action": tfds.features.Tensor(shape=(action_dim,), dtype=np.float32),
                            "discount": tfds.features.Scalar(dtype=np.float32),
                            "reward": tfds.features.Scalar(dtype=np.float32),
                            "is_first": tfds.features.Scalar(dtype=np.bool_),
                            "is_last": tfds.features.Scalar(dtype=np.bool_),
                            "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                            "language_instruction": tfds.features.Text(),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(),
                            "episode_index": tfds.features.Scalar(dtype=np.int32),
                        }
                    ),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        return {"train": self._generate_examples(source_dir=self.SOURCE_DIR)}

    def _generate_examples(self, source_dir: str):
        if self.SOURCE_KIND == "robotwin" and self.EMBODIMENT == "aloha":
            yield from self._generate_aloha(source_dir)
        elif self.SOURCE_KIND == "robotwin":
            yield from self._generate_robotwin(source_dir)
        else:
            yield from self._generate_lerobot(source_dir)

    def _generate_lerobot(self, source_dir: str):
        root = Path(source_dir)
        info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
        tasks = load_tasks(root / "meta")
        episodes = load_episodes(root / "meta")
        chunks_size = int(info.get("chunks_size", 1000))
        data_tmpl = info["data_path"]
        video_tmpl = info["video_path"]
        hw = self.IMAGE_HW

        max_episodes = getattr(self, "MAX_EPISODES", None)
        if max_episodes is not None:
            episodes = episodes[: int(max_episodes)]

        total = len(episodes)
        print(f"converting {total} LeRobot episodes from {root}")
        kept = 0
        skipped = 0
        for idx, ep in enumerate(episodes, start=1):
            ep_i = int(ep["episode_index"])
            parquet_rel = format_lerobot_path(data_tmpl, ep_i, chunks_size)
            parquet_path = root / parquet_rel
            base_rel = format_lerobot_path(video_tmpl, ep_i, chunks_size, "observation.images.base")
            wrist_rel = format_lerobot_path(video_tmpl, ep_i, chunks_size, "observation.images.wrist")
            base_path = root / base_rel
            wrist_path = root / wrist_rel

            try:
                rec = read_parquet_episode(parquet_path)
                t_len = int(rec["state"].shape[0])
                if self.PLACEHOLDER_IMAGES:
                    if not base_path.is_file() or not wrist_path.is_file():
                        print(f"[warn] episode {ep_i}: RGB missing, using black placeholder frames")
                    base = decode_video(base_path, hw) if base_path.is_file() else placeholder_video(t_len, hw)
                    wrist = decode_video(wrist_path, hw) if wrist_path.is_file() else placeholder_video(t_len, hw)
                else:
                    if not base_path.is_file() or not wrist_path.is_file():
                        raise FileNotFoundError(
                            f"RGB video missing for episode {ep_i}. "
                            f"base={base_path.is_file()} wrist={wrist_path.is_file()}. "
                            "Copy observation.images.base / observation.images.wrist mp4 files, "
                            "or pass --placeholder_images to debug without RGB."
                        )
                    base = decode_video(base_path, hw)
                    wrist = decode_video(wrist_path, hw)
            except Exception as exc:  # noqa: BLE001
                skipped += 1
                print(f"[skip] episode {ep_i}: {exc}")
                continue

            n = min(t_len, int(base.shape[0]), int(wrist.shape[0]))
            if n < 2:
                skipped += 1
                print(f"[skip] episode {ep_i}: too few aligned frames ({n})")
                continue
            if n != t_len or n != base.shape[0] or n != wrist.shape[0]:
                print(
                    f"[warn] episode {ep_i}: align to {n} frames "
                    f"(parquet={t_len}, base={base.shape[0]}, wrist={wrist.shape[0]})"
                )

            state = rec["state"][:n]
            action = rec["action"][:n]
            joints = state[:, :7]
            act_joints = action[:, :7]
            state_grip = state[:, 7]
            act_grip = action[:, 7]
            if self.INVERT_GRIPPER:
                act_grip = 1.0 - np.clip(act_grip, 0.0, 1.0)
                state_grip = 1.0 - np.clip(state_grip, 0.0, 1.0)

            proprio = joints_to_proprio(joints, state_grip)
            action_vec = joint_delta_actions(joints, act_joints, act_grip)

            task_idx = int(rec["task_index"][0])
            instruction = tasks.get(task_idx)
            if not instruction:
                names = ep.get("tasks") or []
                instruction = names[0] if names else "complete the task."

            steps = _pack_steps(
                n=n,
                base=base[:n],
                wrist=wrist[:n],
                proprio=proprio,
                action=action_vec,
                joints=joints,
                instruction=instruction,
                skip_noops=self.SKIP_NOOPS,
            )
            kept += 1
            if idx == 1 or idx % 20 == 0 or idx == total:
                print(
                    f"[{idx}/{total}] episode {ep_i}  raw={n} kept={len(steps)}  "
                    f"task={instruction}"
                )

            yield f"episode_{ep_i}", {
                "steps": steps,
                "episode_metadata": {
                    "file_path": str(parquet_path),
                    "episode_index": np.int32(ep_i),
                },
            }

        print(f"done generating: kept={kept} skipped={skipped}")

    def _generate_robotwin(self, source_dir: str):
        import h5py

        source = Path(source_dir)
        data_dir = source / "data"
        inst_dir = find_instruction_dir(source)
        hdf5_files = sorted(data_dir.glob("episode*.hdf5"), key=_episode_id_from_path)
        max_episodes = getattr(self, "MAX_EPISODES", None)
        if max_episodes is not None:
            hdf5_files = hdf5_files[: int(max_episodes)]

        arm = str(self.ARM)
        head_keys = ("vision/cam_head/colors", "observation/head_camera/rgb")
        wrist_keys = (
            (f"vision/cam_{arm}_wrist/colors", f"observation/{arm}_camera/rgb")
            if arm == "left"
            else ("vision/cam_right_wrist/colors", "observation/right_camera/rgb")
        )
        hw = self.IMAGE_HW
        total = len(hdf5_files)
        print(f"converting {total} RoboTwin episodes from {data_dir}  arm={arm}")
        kept = 0
        skipped = 0

        for idx, hdf5_path in enumerate(hdf5_files, start=1):
            episode_id = _episode_id_from_path(hdf5_path)
            inst_path = find_instruction_file(inst_dir, episode_id)
            try:
                with h5py.File(hdf5_path, "r") as h5:
                    rec = read_robotwin_arm(h5, arm)
                    t_len = int(rec["joints"].shape[0])
                    head_ds = next((h5[k] for k in head_keys if k in h5), None)
                    wrist_ds = next((h5[k] for k in wrist_keys if k in h5), None)
                    if head_ds is None or wrist_ds is None:
                        raise KeyError(
                            f"missing RGB for episode {episode_id}: "
                            f"head={head_ds is not None} {arm}_wrist={wrist_ds is not None}"
                        )
                    head_raw = head_ds[:]
                    wrist_raw = wrist_ds[:]
                    hdf5_inst = _as_str(h5["instructions"][()]) if "instructions" in h5 else ""
            except Exception as exc:  # noqa: BLE001
                skipped += 1
                print(f"[skip] episode {episode_id}: {exc}")
                continue

            n = min(t_len, len(head_raw), len(wrist_raw))
            if n < 2:
                skipped += 1
                print(f"[skip] episode {episode_id}: too few aligned frames ({n})")
                continue

            try:
                base = np.stack([_decode_jpeg(head_raw[t], hw) for t in range(n)], axis=0)
                wrist = np.stack([_decode_jpeg(wrist_raw[t], hw) for t in range(n)], axis=0)
            except Exception as exc:  # noqa: BLE001
                skipped += 1
                print(f"[skip] episode {episode_id}: jpeg decode failed ({exc})")
                continue

            joints = rec["joints"][:n]
            state_grip = rec["state_grip"][:n]
            act_grip = rec["act_grip"][:n]
            if self.INVERT_GRIPPER:
                act_grip = 1.0 - np.clip(act_grip, 0.0, 1.0)
                state_grip = 1.0 - np.clip(state_grip, 0.0, 1.0)

            proprio = joints_to_proprio(joints, state_grip)
            action_vec = joint_delta_actions(joints, rec["act_joints"][:n], act_grip)
            instruction = robotwin_instruction(inst_path, episode_id, hdf5_inst)

            steps = _pack_steps(
                n=n,
                base=base,
                wrist=wrist,
                proprio=proprio,
                action=action_vec,
                joints=joints,
                instruction=instruction,
                skip_noops=self.SKIP_NOOPS,
            )
            kept += 1
            if idx == 1 or idx % 50 == 0 or idx == total:
                dq = action_vec[:, :JOINT_DIM]
                print(
                    f"[{idx}/{total}] episode {episode_id}  raw={n} kept={len(steps)}  "
                    f"|dq| mean={float(np.linalg.norm(dq, axis=1).mean()):.4f}  "
                    f"task={instruction}"
                )

            yield f"episode_{episode_id}", {
                "steps": steps,
                "episode_metadata": {
                    "file_path": str(hdf5_path),
                    "episode_index": np.int32(episode_id),
                },
            }

        print(f"done generating: kept={kept} skipped={skipped}")

    def _generate_aloha(self, source_dir: str):
        import h5py

        source = Path(source_dir)
        data_dir = source / "data"
        inst_dir = find_instruction_dir(source)
        hdf5_files = sorted(data_dir.glob("episode*.hdf5"), key=_episode_id_from_path)
        max_episodes = getattr(self, "MAX_EPISODES", None)
        if max_episodes is not None:
            hdf5_files = hdf5_files[: int(max_episodes)]

        cam_keys = {
            "image": ("vision/cam_head/colors", "observation/head_camera/rgb"),
            "left_wrist_image": ("vision/cam_left_wrist/colors", "observation/left_camera/rgb"),
            "right_wrist_image": ("vision/cam_right_wrist/colors", "observation/right_camera/rgb"),
        }
        hw = self.IMAGE_HW
        total = len(hdf5_files)
        print(f"converting {total} Aloha episodes from {data_dir}")
        kept = 0
        skipped = 0

        for idx, hdf5_path in enumerate(hdf5_files, start=1):
            episode_id = _episode_id_from_path(hdf5_path)
            inst_path = find_instruction_file(inst_dir, episode_id)
            try:
                with h5py.File(hdf5_path, "r") as h5:
                    arms = {}
                    for side in ("left", "right"):
                        joints = np.asarray(h5[f"state/{side}_arm_joint_states"][:], dtype=np.float32)
                        act_joints = np.asarray(h5[f"action/{side}_arm_joint_states"][:], dtype=np.float32)
                        state_grip = np.asarray(h5[f"state/{side}_ee_joint_states"][:], dtype=np.float32).reshape(-1)
                        act_grip = np.asarray(h5[f"action/{side}_ee_joint_states"][:], dtype=np.float32).reshape(-1)
                        if joints.ndim != 2 or joints.shape[1] < ALOHA_JOINT_DIM:
                            raise ValueError(f"{side} joints shape {joints.shape}, expected (T, {ALOHA_JOINT_DIM})")
                        arms[side] = (joints[:, :ALOHA_JOINT_DIM], act_joints[:, :ALOHA_JOINT_DIM], state_grip, act_grip)
                    raw_cams = {}
                    for name, keys in cam_keys.items():
                        ds = next((h5[k] for k in keys if k in h5), None)
                        if ds is None:
                            raise KeyError(f"missing {name} RGB for episode {episode_id}")
                        raw_cams[name] = ds[:]
                    hdf5_inst = _as_str(h5["instructions"][()]) if "instructions" in h5 else ""
            except Exception as exc:  # noqa: BLE001
                skipped += 1
                print(f"[skip] episode {episode_id}: {exc}")
                continue

            lengths = [int(arr.shape[0]) for arm in arms.values() for arr in arm]
            lengths.extend(len(raw) for raw in raw_cams.values())
            n = min(lengths)
            if n < 2:
                skipped += 1
                print(f"[skip] episode {episode_id}: too few aligned frames ({n})")
                continue

            try:
                images = {
                    name: np.stack([_decode_jpeg(raw[t], hw) for t in range(n)], axis=0)
                    for name, raw in raw_cams.items()
                }
            except Exception as exc:  # noqa: BLE001
                skipped += 1
                print(f"[skip] episode {episode_id}: jpeg decode failed ({exc})")
                continue

            left_q, left_aq, left_sg, left_ag = arms["left"]
            right_q, right_aq, right_sg, right_ag = arms["right"]
            left_sg, left_ag = left_sg[:n], left_ag[:n]
            right_sg, right_ag = right_sg[:n], right_ag[:n]
            if self.INVERT_GRIPPER:
                left_sg = 1.0 - np.clip(left_sg, 0.0, 1.0)
                left_ag = 1.0 - np.clip(left_ag, 0.0, 1.0)
                right_sg = 1.0 - np.clip(right_sg, 0.0, 1.0)
                right_ag = 1.0 - np.clip(right_ag, 0.0, 1.0)

            state = pack_aloha_vector(left_q[:n], left_sg, right_q[:n], right_sg)
            action = pack_aloha_vector(left_aq[:n], left_ag, right_aq[:n], right_ag)
            instruction = robotwin_instruction(inst_path, episode_id, hdf5_inst)

            keep = np.ones(n, dtype=bool)
            if self.SKIP_NOOPS:
                dq = np.concatenate([action[:, 0:6] - state[:, 0:6], action[:, 7:13] - state[:, 7:13]], axis=1)
                keep = np.linalg.norm(dq, axis=1) >= NOOP_THRESH
                keep[0] = True
                keep[-1] = True
                if int(keep.sum()) < 2:
                    keep[:] = True
            indices = np.flatnonzero(keep)
            last_k = int(indices[-1])
            steps = []
            for step_i, t in enumerate(indices):
                is_last = int(t) == last_k
                steps.append(
                    {
                        "observation": {
                            "image": images["image"][t],
                            "left_wrist_image": images["left_wrist_image"][t],
                            "right_wrist_image": images["right_wrist_image"][t],
                            "state": state[t],
                        },
                        "action": action[t],
                        "discount": np.float32(1.0),
                        "reward": np.float32(1.0 if is_last else 0.0),
                        "is_first": step_i == 0,
                        "is_last": is_last,
                        "is_terminal": is_last,
                        "language_instruction": instruction,
                    }
                )

            kept += 1
            if idx == 1 or idx % 50 == 0 or idx == total:
                print(
                    f"[{idx}/{total}] episode {episode_id}  raw={n} kept={len(steps)}  "
                    f"task={instruction}"
                )

            yield f"episode_{episode_id}", {
                "steps": steps,
                "episode_metadata": {
                    "file_path": str(hdf5_path),
                    "episode_index": np.int32(episode_id),
                },
            }

        print(f"done generating: kept={kept} skipped={skipped}")


def _tfds_class_name(dataset_name: str) -> str:
    """CamelCase that round-trips through TFDS snake_case.

    ``grasp_conveyor_rand_speed_2``.split + capitalize becomes
    ``GraspConveyorRandSpeed2``, which TFDS writes as
    ``grasp_conveyor_rand_speed2``. Keep a trailing ``_N`` on the last word.
    """
    chunks: list[str] = []
    for part in dataset_name.split("_"):
        if part.isdigit():
            if not chunks:
                raise ValueError(f"dataset name cannot start with a number: {dataset_name}")
            chunks[-1] = f"{chunks[-1]}_{part}"
        else:
            chunks.append(part.capitalize())
    return "".join(chunks)


def _builder_class(dataset_name: str):
    if dataset_name == ByangPickUp.name:
        ByangPickUp.DATASET_DESCRIPTION = f"{dataset_name} converted to 8-D joint-delta RLDS."
        return ByangPickUp

    class_name = _tfds_class_name(dataset_name)
    existing = globals().get(class_name)
    if existing is not None:
        return existing
    return type(
        class_name,
        (ByangPickUp,),
        {"DATASET_DESCRIPTION": f"{dataset_name} converted to 8-D joint-delta RLDS."},
    )


def _print_lerobot_report(root: Path) -> dict:
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    n_ep = int(info["total_episodes"])
    n_base = count_videos(root, info, "observation.images.base")
    n_wrist = count_videos(root, info, "observation.images.wrist")
    print(f"source dataset : {root.name}")
    print(f"  kind         : LeRobot v2.1")
    print(f"  robot        : {info.get('robot_type')}")
    print(f"  episodes     : {n_ep}")
    print(f"  frames       : {info.get('total_frames')}")
    print(f"  tasks        : {info.get('total_tasks')}")
    print(f"  fps          : {info.get('fps')}")
    print(f"  RGB base     : {n_base}/{n_ep}")
    print(f"  RGB wrist    : {n_wrist}/{n_ep}")
    return {"info": info, "n_base": n_base, "n_wrist": n_wrist, "n_ep": n_ep}


def _print_robotwin_report(root: Path, arm: str, embodiment: str = "franka") -> dict:
    data_dir = root / "data"
    hdf5_files = sorted(data_dir.glob("episode*.hdf5"), key=_episode_id_from_path)
    n_ep = len(hdf5_files)
    print(f"source dataset : {root}")
    print(f"  kind         : RoboTwin hdf5")
    print(f"  embodiment   : {embodiment}")
    print(f"  episodes     : {n_ep}")
    if embodiment == "aloha":
        print("  arms         : left + right (6 joints + gripper each)")
        print("  images       : cam_head + cam_left_wrist + cam_right_wrist inside hdf5")
    else:
        print(f"  arm          : {arm} only (other arm parked)")
        print(f"  images       : cam_head + cam_{arm}_wrist inside hdf5")
    return {"n_ep": n_ep, "n_base": n_ep, "n_wrist": n_ep}


def convert_one(
    source_dir: Path,
    output_dir: Path,
    dataset_name: str,
    source_kind: str,
    overwrite: bool,
    max_episodes: Optional[int],
    skip_noops: bool,
    placeholder_images: bool,
    invert_gripper: bool,
    arm: str,
    embodiment: str = "franka",
) -> None:
    if source_kind == "lerobot":
        report = _print_lerobot_report(source_dir)
        if report["n_base"] == 0 or report["n_wrist"] == 0:
            if not placeholder_images:
                raise FileNotFoundError(
                    "RGB videos are missing.\n"
                    f"  expected: {source_dir}/videos/chunk-*/observation.images.base/episode_XXXXXX.mp4\n"
                    f"  expected: {source_dir}/videos/chunk-*/observation.images.wrist/episode_XXXXXX.mp4\n"
                    "Currently only depth mp4 files were found. Copy the RGB videos into those folders, "
                    "then rerun. For a pipeline smoke test you can pass --placeholder_images."
                )
            print("[warn] RGB videos missing; writing black placeholder images")
    else:
        find_instruction_dir(source_dir)
        _print_robotwin_report(source_dir, arm, embodiment)

    ds_dir = output_dir / dataset_name / "1.0.0"
    if ds_dir.is_dir() and not overwrite:
        print(f"RLDS already exists at {ds_dir}")
        print("Pass --overwrite to rebuild.")
        return

    if overwrite:
        built = output_dir / dataset_name
        if built.exists():
            shutil.rmtree(built)

    builder_cls = _builder_class(dataset_name)
    builder_cls.SOURCE_DIR = str(source_dir)
    builder_cls.SOURCE_KIND = source_kind
    builder_cls.MAX_EPISODES = max_episodes
    builder_cls.SKIP_NOOPS = skip_noops
    builder_cls.PLACEHOLDER_IMAGES = placeholder_images
    builder_cls.INVERT_GRIPPER = invert_gripper
    builder_cls.IMAGE_HW = IMAGE_HW
    builder_cls.ARM = arm
    builder_cls.EMBODIMENT = embodiment
    if embodiment == "aloha":
        builder_cls.DATASET_DESCRIPTION = f"{dataset_name} converted to 14-D Aloha absolute-joint RLDS."
        action_desc = f"{ALOHA_ACTION_DIM}-dim absolute joints (left q6+grip + right q6+grip)"
        state_desc = f"{ALOHA_STATE_DIM}-dim joints + grippers"
        image_desc = f"head -> image, both wrists, resize {IMAGE_HW}"
    else:
        action_desc = f"{ACTION_DIM}-dim joint delta (Δq7 + gripper)"
        state_desc = f"{STATE_DIM}-dim joints + gripper"
        image_desc = f"head/base -> image, {arm} wrist -> wrist_image, resize {IMAGE_HW}"

    print(f"output         : {output_dir}")
    print(f"dataset        : {dataset_name}")
    print(f"action         : {action_desc}")
    print(f"state          : {state_desc}")
    print(f"images         : {image_desc}")
    print(f"skip_noops     : {skip_noops}")
    print(f"invert_gripper : {invert_gripper}")

    builder = builder_cls(data_dir=str(output_dir))
    download_mode = (
        tfds.GenerateMode.FORCE_REDOWNLOAD if overwrite else tfds.GenerateMode.REUSE_DATASET_IF_EXISTS
    )
    builder.download_and_prepare(
        download_config=tfds.download.DownloadConfig(download_mode=download_mode),
        file_format="tfrecord",
    )
    print(f"done: {ds_dir}")
    print("Train with:")
    print(f"  --data_root_dir {output_dir}")
    print(f"  --dataset_name {dataset_name}")
    if embodiment == "aloha":
        print("  --num_images_in_input 3 --use_proprio True")
    else:
        print("  --num_images_in_input 2 --use_proprio True")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert LeRobot, RoboTwin Franka, or RoboTwin Aloha hdf5 demos to RLDS"
    )
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="TFDS dataset name. Default: LeRobot folder name or RoboTwin task name.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument(
        "--skip_noops",
        action="store_true",
        help="Drop near-zero joint-delta steps.",
    )
    parser.add_argument(
        "--placeholder_images",
        action="store_true",
        help="Use black frames when RGB mp4 files are missing (LeRobot debug only).",
    )
    parser.add_argument(
        "--invert_gripper",
        action="store_true",
        help="Flip gripper 0/1. Default ON for RoboTwin (native 1=open) so "
        "franka_joint_dataset_transform yields 1=open after the extra invert.",
    )
    parser.add_argument(
        "--no_invert_gripper",
        action="store_true",
        help="Keep native gripper (RoboTwin 1=open). Use if open/close looks inverted.",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default=DEFAULT_ARM,
        choices=("right", "left"),
        help="Which RoboTwin arm to export. Default: right (left is parked).",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    roots = discover_dataset_roots(args.source_dir)
    for kind, root, default_name in roots:
        name = args.dataset_name or default_name
        embodiment = "franka"
        if kind == "lerobot":
            _require_pyarrow()
            invert = bool(args.invert_gripper) and not args.no_invert_gripper
        else:
            embodiment = detect_robotwin_embodiment(root)
            if embodiment == "aloha":
                # aloha_dataset_transform does not invert the gripper a second time.
                invert = bool(args.invert_gripper) and not args.no_invert_gripper
            else:
                invert = not args.no_invert_gripper
        print("=" * 72)
        convert_one(
            source_dir=root,
            output_dir=output_dir,
            dataset_name=name,
            source_kind=kind,
            overwrite=args.overwrite,
            max_episodes=args.max_episodes,
            skip_noops=args.skip_noops,
            placeholder_images=args.placeholder_images,
            invert_gripper=invert,
            arm=args.arm,
            embodiment=embodiment,
        )


if __name__ == "__main__":
    tf.config.set_visible_devices([], "GPU")
    main()
