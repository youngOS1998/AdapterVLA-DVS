#!/usr/bin/env python3
"""Inspect a TFDS / RLDS dataset and print what action / state represent.

Example::

    /root/anaconda3/envs/vla-adapter/bin/python \\
        data_process/inspect_rlds.py \\
        --data_dir /mnt/Projects_origin/VLA-Adapter/data/robotwin_rlds/place_color_count_blocks_tray_rand \\
        --num_episodes 4
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("NO_GCE_CHECK", "true")

import numpy as np
import tensorflow_datasets as tfds
from tensorflow_datasets.core.utils import gcs_utils

gcs_utils._is_gcs_disabled = True


ACTION_LABELS = {
    7: [
        "Δx  (EEF, meters)",
        "Δy  (EEF, meters)",
        "Δz  (EEF, meters)",
        "Δroll  (EEF, rad)",
        "Δpitch (EEF, rad)",
        "Δyaw   (EEF, rad)",
        "gripper  (this dump: 0=open, 1=close if RoboTwin invert_gripper; after libero_dataset_transform: 1=open, 0=close)",
    ],
    8: [
        "Δq1 (rad)",
        "Δq2 (rad)",
        "Δq3 (rad)",
        "Δq4 (rad)",
        "Δq5 (rad)",
        "Δq6 (rad)",
        "Δq7 (rad)",
        "gripper",
    ],
    14: [
        "left_j1",
        "left_j2",
        "left_j3",
        "left_j4",
        "left_j5",
        "left_j6",
        "left_gripper",
        "right_j1",
        "right_j2",
        "right_j3",
        "right_j4",
        "right_j5",
        "right_j6",
        "right_gripper",
    ],
    16: [
        "left_j1",
        "left_j2",
        "left_j3",
        "left_j4",
        "left_j5",
        "left_j6",
        "left_j7",
        "left_gripper",
        "right_j1",
        "right_j2",
        "right_j3",
        "right_j4",
        "right_j5",
        "right_j6",
        "right_j7",
        "right_gripper",
    ],
}

STATE_LABELS_7D_EEF = [
    "x (m)",
    "y (m)",
    "z (m)",
    "roll (rad)",
    "pitch (rad)",
    "yaw (rad)",
    "gripper_a",
    "gripper_b",
]


def _as_str(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _load_stats(data_dir: Path) -> dict | None:
    files = sorted(data_dir.glob("dataset_statistics_*.json"))
    if not files:
        return None
    with files[0].open("r", encoding="utf-8") as f:
        return json.load(f)


def _describe_action(dim: int) -> str:
    if dim == 7:
        return "LIBERO-style 7-D: right-arm EEF delta xyz + rpy + gripper (NOT joint angles)"
    if dim == 8:
        return "single-arm 8-D: 7 joint-angle deltas (rad) + gripper"
    if dim == 14:
        return "ALOHA 14-D: left 6+grip + right 6+grip absolute joints"
    if dim == 16:
        return "dual Franka 16-D: left 7+grip + right 7+grip absolute joints"
    return f"unknown action layout, dim={dim}"


def inspect(data_dir: Path, num_episodes: int) -> None:
    version_dir = data_dir / "1.0.0" if (data_dir / "1.0.0").is_dir() else data_dir
    builder = tfds.builder_from_directory(str(version_dir))
    info = builder.info
    print("=" * 72)
    print(f"path     : {version_dir}")
    print(f"name     : {info.name}")
    print(f"version  : {info.version}")
    print(f"splits   : { {k: int(v.num_examples) for k, v in info.splits.items()} }")
    print(f"features :\n{info.features}")

    stats = _load_stats(version_dir)
    if stats:
        print("-" * 72)
        print("dataset_statistics (computed after dataloader transform if present)")
        print(f"  trajectories : {stats.get('num_trajectories')}")
        print(f"  transitions  : {stats.get('num_transitions')}")
        for key in ("action", "proprio"):
            if key not in stats:
                continue
            block = stats[key]
            print(f"  {key}.mean : {np.round(block['mean'], 5).tolist()}")
            print(f"  {key}.min  : {np.round(block['min'], 5).tolist()}")
            print(f"  {key}.max  : {np.round(block['max'], 5).tolist()}")

    ds = builder.as_dataset(split="train", shuffle_files=False)
    langs: Counter[str] = Counter()
    lengths = []

    print("-" * 72)
    for ep_i, ep in enumerate(ds.take(num_episodes)):
        steps = list(ep["steps"])
        t_len = len(steps)
        lengths.append(t_len)
        lang = _as_str(steps[0]["language_instruction"].numpy())
        langs[lang] += 1
        meta = ep.get("episode_metadata", {})
        file_path = _as_str(meta["file_path"].numpy()) if "file_path" in meta else ""
        actions = np.stack([st["action"].numpy() for st in steps])
        state = np.stack([st["observation"]["state"].numpy() for st in steps])
        obs = steps[0]["observation"]
        joint = None
        if "joint_state" in obs:
            joint = np.stack([st["observation"]["joint_state"].numpy() for st in steps])

        print(f"\n==== episode {ep_i}  T={t_len} ====")
        if file_path:
            print(f"file_path : {file_path}")
        print(f"language  : {lang}")
        print(f"action    : shape={actions.shape}  {_describe_action(actions.shape[-1])}")
        labels = ACTION_LABELS.get(int(actions.shape[-1]), [f"dim_{i}" for i in range(actions.shape[-1])])
        print(f"action[0] : {np.round(actions[0], 5).tolist()}")
        print(f"action[-1]: {np.round(actions[-1], 5).tolist()}")
        print(f"action min: {np.round(actions.min(0), 5).tolist()}")
        print(f"action max: {np.round(actions.max(0), 5).tolist()}")
        print("per-dim range / meaning:")
        for i, name in enumerate(labels):
            print(
                f"  [{i}] {name:28s}  "
                f"mean={actions[:, i].mean():+.5f}  "
                f"std={actions[:, i].std():.5f}  "
                f"[{actions[:, i].min():+.5f}, {actions[:, i].max():+.5f}]"
            )
        print(f"state[0]  : {np.round(state[0], 5).tolist()}")
        if state.shape[-1] == 8:
            if actions.shape[-1] == 8:
                print("state     : 8-D current joints q1..q7 + gripper")
                state_names = [f"q{i+1} (rad)" for i in range(7)] + ["gripper"]
            else:
                print("state     : 8-D EEF xyz + rpy + [grip, grip]")
                state_names = STATE_LABELS_7D_EEF
            for i, name in enumerate(state_names):
                print(f"  [{i}] {name:16s}  mean={state[:, i].mean():+.5f}")
        if joint is not None:
            print(f"joint[0]  : {np.round(joint[0], 5).tolist()}  (7 Franka joints, rad)")
            print(
                "joint motion range:",
                np.round(joint.max(0) - joint.min(0), 5).tolist(),
            )
        img = obs["image"].numpy()
        print(f"image     : {img.shape} {img.dtype}", end="")
        if "wrist_image" in obs:
            print(f"  wrist={obs['wrist_image'].numpy().shape}", end="")
        if "left_wrist_image" in obs:
            print(f"  left_wrist={obs['left_wrist_image'].numpy().shape}", end="")
        if "right_wrist_image" in obs:
            print(f"  right_wrist={obs['right_wrist_image'].numpy().shape}", end="")
        print()
        print(
            f"flags     : first={bool(steps[0]['is_first'].numpy())}  "
            f"last={bool(steps[-1]['is_last'].numpy())}  "
            f"terminal={bool(steps[-1]['is_terminal'].numpy())}  "
            f"reward_last={float(steps[-1]['reward'].numpy())}"
        )
        if actions.shape[-1] == 7:
            dxyz = np.linalg.norm(actions[:, :3], axis=1)
            drpy = np.linalg.norm(actions[:, 3:6], axis=1)
            print(f"||Δxyz|| m  mean/max = {dxyz.mean():.5f} / {dxyz.max():.5f}")
            print(f"||Δrpy|| rad mean/max = {drpy.mean():.5f} / {drpy.max():.5f}")
            print(f"gripper unique (rounded 3dp, first 12): {np.unique(np.round(actions[:, -1], 3))[:12]}")

    print("\n" + "=" * 72)
    print(f"previewed {len(lengths)} episodes, lengths min/mean/max = {min(lengths)} / {float(np.mean(lengths)):.1f} / {max(lengths)}")
    print("language counts in preview:")
    for text, n in langs.most_common():
        print(f"  [{n:3d}] {text}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect RLDS action / observation layout")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("/mnt/Projects_origin/VLA-Adapter/data/robotwin_rlds/place_color_count_blocks_tray_rand"),
    )
    parser.add_argument("--num_episodes", type=int, default=4)
    args = parser.parse_args()
    inspect(args.data_dir.expanduser().resolve(), args.num_episodes)


if __name__ == "__main__":
    main()
