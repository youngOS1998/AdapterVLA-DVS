#!/usr/bin/env python3
"""Visualize one RLDS episode: camera strip, joint curves, and a side-by-side mp4.

Works for both Aloha 14-D (3 cameras) and single-arm Franka 8-D (2 cameras).

Example::

    /root/anaconda3/envs/vla-adapter/bin/python \\
        data_process/visualize_rlds_episode.py \\
        --data_dir /mnt/Projects_origin/VLA-Adapter/data/robotwin_rlds/grasp_conveyor_block
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("NO_GCE_CHECK", "true")
os.environ.setdefault("MPLBACKEND", "Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image, ImageDraw, ImageFont
from tensorflow_datasets.core.utils import gcs_utils

gcs_utils._is_gcs_disabled = True

ALOHA_NAMES = [
    "Lq1",
    "Lq2",
    "Lq3",
    "Lq4",
    "Lq5",
    "Lq6",
    "Lgrip",
    "Rq1",
    "Rq2",
    "Rq3",
    "Rq4",
    "Rq5",
    "Rq6",
    "Rgrip",
]
FRANKA_NAMES = ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "grip"]


def _as_str(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _version_dir(data_dir: Path) -> Path:
    if (data_dir / "1.0.0").is_dir():
        return data_dir / "1.0.0"
    return data_dir


def load_episode(data_dir: Path, episode_index: int | None):
    builder = tfds.builder_from_directory(str(_version_dir(data_dir)))
    ds = builder.as_dataset(split="train", shuffle_files=False)
    chosen = None
    for ep in ds:
        meta = ep["episode_metadata"]
        idx = int(meta["episode_index"].numpy())
        if episode_index is None or idx == episode_index:
            chosen = ep
            break
    if chosen is None:
        raise FileNotFoundError(f"episode_index={episode_index} not found under {data_dir}")

    steps = list(chosen["steps"])
    obs0 = steps[0]["observation"]
    cameras = ["image"]
    for name in ("left_wrist_image", "right_wrist_image", "wrist_image"):
        if name in obs0:
            cameras.append(name)

    images = {
        name: np.stack([step["observation"][name].numpy() for step in steps])
        for name in cameras
    }
    state = np.stack([step["observation"]["state"].numpy() for step in steps]).astype(np.float32)
    action = np.stack([step["action"].numpy() for step in steps]).astype(np.float32)
    meta = chosen["episode_metadata"]
    info = {
        "episode_index": int(meta["episode_index"].numpy()),
        "file_path": _as_str(meta["file_path"].numpy()) if "file_path" in meta else "",
        "language": _as_str(steps[0]["language_instruction"].numpy()),
        "reward_last": float(steps[-1]["reward"].numpy()),
        "is_first": bool(steps[0]["is_first"].numpy()),
        "is_last": bool(steps[-1]["is_last"].numpy()),
    }
    return images, state, action, info


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def save_camera_strip(images: dict[str, np.ndarray], info: dict, out_path: Path, n_cols: int) -> None:
    names = list(images.keys())
    t_len = images[names[0]].shape[0]
    cols = min(n_cols, t_len)
    times = np.unique(np.linspace(0, t_len - 1, cols, dtype=int))
    cell_h, cell_w = images[names[0]].shape[1:3]
    label_h = 28
    header_h = 36
    canvas = Image.new("RGB", (cell_w * len(times), header_h + (cell_h + label_h) * len(names)), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    font = _font(16)
    small = _font(14)
    title = f"ep {info['episode_index']}  T={t_len}  {info['language']}"
    draw.text((8, 8), title[:140], fill=(240, 240, 240), font=font)

    for row, name in enumerate(names):
        y0 = header_h + row * (cell_h + label_h)
        draw.text((6, y0 + 4), name, fill=(255, 220, 120), font=small)
        for col, t in enumerate(times):
            frame = Image.fromarray(images[name][int(t)])
            canvas.paste(frame, (col * cell_w, y0 + label_h))
            draw.text((col * cell_w + 6, y0 + label_h + 4), f"t={int(t)}", fill=(255, 255, 255), font=small)
    canvas.save(out_path)


def _dim_names(dim: int) -> list[str]:
    if dim == 14:
        return ALOHA_NAMES
    if dim == 8:
        return FRANKA_NAMES
    return [f"d{i}" for i in range(dim)]


def save_joint_plot(state: np.ndarray, action: np.ndarray, info: dict, out_path: Path) -> None:
    dim = int(state.shape[-1])
    names = _dim_names(dim)
    t = np.arange(state.shape[0])
    ncols = 7 if dim >= 7 else dim
    nrows = int(np.ceil(dim / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.2 * nrows), sharex=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    for i in range(dim):
        ax = axes[i // ncols, i % ncols]
        ax.plot(t, state[:, i], color="#1f77b4", lw=1.4, label="state")
        ax.plot(t, action[:, i], color="#d62728", lw=1.0, ls="--", label="action")
        ax.set_title(names[i], fontsize=10)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8, loc="best")
    for i in range(dim, nrows * ncols):
        axes[i // ncols, i % ncols].axis("off")
    fig.suptitle(
        f"episode {info['episode_index']}  state vs action  |  {info['language']}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def save_video(images: dict[str, np.ndarray], info: dict, out_path: Path, fps: int) -> None:
    names = list(images.keys())
    frames = [images[name] for name in names]
    t_len, height, width, _ = frames[0].shape
    gap = 4
    canvas_w = width * len(names) + gap * (len(names) - 1)
    canvas_h = height + 28
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (canvas_w, canvas_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer for {out_path}")
    font = _font(16)
    for t in range(t_len):
        strip = Image.new("RGB", (canvas_w, canvas_h), (20, 20, 20))
        draw = ImageDraw.Draw(strip)
        draw.text(
            (8, 4),
            f"ep {info['episode_index']}  t={t}/{t_len - 1}  {info['language'][:90]}",
            fill=(240, 240, 240),
            font=font,
        )
        for i, frame in enumerate(frames):
            strip.paste(Image.fromarray(frame[t]), (i * (width + gap), 28))
        bgr = cv2.cvtColor(np.asarray(strip), cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()


def print_summary(images: dict[str, np.ndarray], state: np.ndarray, action: np.ndarray, info: dict) -> None:
    names = _dim_names(int(state.shape[-1]))
    print("=" * 72)
    print(f"episode_index : {info['episode_index']}")
    print(f"file_path     : {info['file_path']}")
    print(f"language      : {info['language']}")
    print(f"T             : {state.shape[0]}")
    print(f"state/action  : {state.shape} / {action.shape}")
    print(f"flags         : is_first={info['is_first']} is_last={info['is_last']} reward_last={info['reward_last']}")
    for name, arr in images.items():
        print(f"{name:18s}: {arr.shape} {arr.dtype}  mean={arr.mean():.1f}")
    if "left_wrist_image" in images and "right_wrist_image" in images:
        diff = np.abs(images["left_wrist_image"].astype(np.int16) - images["right_wrist_image"].astype(np.int16))
        print(f"left vs right wrist mean abs diff: {diff.mean():.2f}")
    if state.shape[-1] == 14:
        left_q, right_q = state[:, 0:6], state[:, 7:13]
        print(f"left vs right joint max abs diff: {np.max(np.abs(left_q - right_q)):.6f}")
        print(
            f"grippers state L/R range: "
            f"[{state[:, 6].min():.3f}, {state[:, 6].max():.3f}] / "
            f"[{state[:, 13].min():.3f}, {state[:, 13].max():.3f}]"
        )
    print("per-dim state mean/std  |  action mean/std")
    for i, name in enumerate(names):
        print(
            f"  [{i:2d}] {name:6s}  "
            f"state {state[:, i].mean():+.4f} ± {state[:, i].std():.4f}  "
            f"action {action[:, i].mean():+.4f} ± {action[:, i].std():.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize one RLDS episode")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("/mnt/Projects_origin/VLA-Adapter/data/robotwin_rlds/grasp_conveyor_block"),
    )
    parser.add_argument("--episode_index", type=int, default=None, help="Default: the first episode in the split.")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--n_cols", type=int, default=8, help="How many timesteps to put on the camera strip.")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    output_dir = (args.output_dir or (data_dir / "vis")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    images, state, action, info = load_episode(data_dir, args.episode_index)
    print_summary(images, state, action, info)

    tag = f"episode_{info['episode_index']:07d}"
    strip_path = output_dir / f"{tag}_cameras.png"
    plot_path = output_dir / f"{tag}_joints.png"
    video_path = output_dir / f"{tag}_preview.mp4"
    save_camera_strip(images, info, strip_path, args.n_cols)
    save_joint_plot(state, action, info, plot_path)
    save_video(images, info, video_path, args.fps)
    print("-" * 72)
    print(f"cameras : {strip_path}")
    print(f"joints  : {plot_path}")
    print(f"video   : {video_path}")


if __name__ == "__main__":
    main()
