"""Export RGB camera streams from RoboTwin-style episode HDF5 files to mp4.

Each episode stores JPEG frames under ``vision/<camera>/colors``. This script
decodes those frames and writes a video so the camera viewpoint can be checked
by eye.

Example:
    python scripts/visualize_camera_view.py \
        --data-dir data/grasp_conveyor_block \
        --camera wrist \
        --episodes 0,13,19
"""

import argparse
import io
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw


def find_ffmpeg() -> tuple[str, str]:
    """Return (ffmpeg_path, video_encoder). Prefer libx264, then libopenh264, then mpeg4."""
    candidates = []
    found = shutil.which("ffmpeg")
    if found:
        candidates.append(found)
    candidates.extend(
        [
            "/root/anaconda3/envs/openvla/bin/ffmpeg",
            "/root/anaconda3/envs/RoboTwin/bin/ffmpeg",
            "/root/anaconda3/envs/ip2p/bin/ffmpeg",
        ]
    )
    seen = []
    for path in candidates:
        if path not in seen and Path(path).is_file():
            seen.append(path)
    for path in seen:
        proc = subprocess.run([path, "-hide_banner", "-encoders"], capture_output=True, text=True)
        encoders = proc.stdout
        for name in ("libx264", "libopenh264", "mpeg4"):
            if name in encoders:
                return path, name
    raise FileNotFoundError("找不到带 H.264/MPEG-4 编码器的 ffmpeg。")


def find_episode_dir(data_dir: Path) -> Path:
    direct = data_dir / "data"
    if direct.is_dir() and any(direct.glob("episode_*.hdf5")):
        return direct
    matches = sorted(p for p in data_dir.rglob("data") if p.is_dir() and any(p.glob("episode_*.hdf5")))
    if not matches:
        raise FileNotFoundError(f"在 {data_dir} 下没有找到 episode_*.hdf5")
    return matches[0]


def list_cameras(hdf5_path: Path) -> list[str]:
    with h5py.File(hdf5_path, "r") as f:
        if "vision" not in f:
            return []
        return sorted(k for k in f["vision"].keys() if "colors" in f["vision"][k])


def resolve_camera(available: list[str], requested: str) -> str | None:
    if requested in available:
        return requested
    needle = requested.lower()
    hits = [name for name in available if needle in name.lower()]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise ValueError(f"相机名 {requested!r} 匹配到多个: {hits}")
    return None


def load_frames(hdf5_path: Path, camera: str) -> tuple[list[Image.Image], int]:
    with h5py.File(hdf5_path, "r") as f:
        blobs = f[f"vision/{camera}/colors"]
        freq = int(f["additional_info/frequency"][()]) if "additional_info/frequency" in f else 30
        frames = []
        for blob in blobs:
            raw = blob.tobytes() if isinstance(blob, np.ndarray) else bytes(blob)
            frames.append(Image.open(io.BytesIO(raw)).convert("RGB"))
    return frames, freq


def annotate(frame: Image.Image, text: str) -> Image.Image:
    canvas = frame.copy()
    draw = ImageDraw.Draw(canvas)
    bar_h = 18
    draw.rectangle((0, 0, canvas.width, bar_h), fill=(0, 0, 0))
    draw.text((4, 2), text, fill=(255, 255, 255))
    return canvas


def write_mp4(frames: list[Image.Image], out_path: Path, fps: int, ffmpeg: str, encoder: str) -> None:
    if not frames:
        raise ValueError("没有可写入的帧")
    width, height = frames[0].size
    # yuv420p needs even dimensions
    if width % 2 or height % 2:
        width -= width % 2
        height -= height % 2
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        encoder,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for frame in frames:
            if frame.size != (width, height):
                frame = frame.resize((width, height))
            proc.stdin.write(frame.tobytes())
    except BrokenPipeError:
        err = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        raise RuntimeError(f"ffmpeg 提前退出:\n{err}") from None
    finally:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
        err = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg 退出码 {code}:\n{err}")


def parse_episodes(text: str, n_available: int) -> list[int]:
    if text.strip().lower() == "all":
        return list(range(n_available))
    ids = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            ids.extend(range(int(start), int(end) + 1))
        else:
            ids.append(int(part))
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="把 HDF5 里的相机 RGB 导出成 mp4，用来查看视角。")
    parser.add_argument("--data-dir", type=Path, required=True, help="数据集根目录，例如 data/grasp_conveyor_block")
    parser.add_argument(
        "--camera",
        default="wrist",
        help="相机名或关键字。默认 wrist，会匹配名字里带 wrist 的相机。",
    )
    parser.add_argument("--episodes", default="0,13,19", help="episode 编号，逗号分隔，也支持 0-4 或 all")
    parser.add_argument("--out", type=Path, default=None, help="输出 mp4 路径")
    parser.add_argument("--fps", type=int, default=None, help="输出帧率，默认用 HDF5 里的 frequency")
    args = parser.parse_args()

    episode_dir = find_episode_dir(args.data_dir)
    hdf5_paths = sorted(episode_dir.glob("episode_*.hdf5"))
    if not hdf5_paths:
        raise FileNotFoundError(f"{episode_dir} 里没有 episode hdf5")

    available = list_cameras(hdf5_paths[0])
    print(f"数据目录: {episode_dir}")
    print(f"共 {len(hdf5_paths)} 条 episode，相机: {available or '无'}")

    camera = resolve_camera(available, args.camera)
    if camera is None:
        print(f"没有找到名字包含 {args.camera!r} 的相机。")
        if not available:
            raise SystemExit(1)
        camera = available[0]
        print(f"改为导出实际存在的相机: {camera}")

    episode_ids = parse_episodes(args.episodes, len(hdf5_paths))
    out_path = args.out or (args.data_dir / "camera_preview" / f"{camera}_preview.mp4")
    ffmpeg, encoder = find_ffmpeg()

    stitched: list[Image.Image] = []
    fps = args.fps
    for episode_id in episode_ids:
        matches = list(episode_dir.glob(f"episode_{episode_id:07d}.hdf5"))
        if not matches:
            print(f"跳过 episode {episode_id}: 文件不存在")
            continue
        frames, file_fps = load_frames(matches[0], camera)
        fps = fps or file_fps
        print(f"episode {episode_id}: {len(frames)} 帧, {frames[0].size[0]}x{frames[0].size[1]}, {file_fps} Hz")
        for i, frame in enumerate(frames):
            stitched.append(annotate(frame, f"{camera}  ep{episode_id:04d}  {i + 1}/{len(frames)}"))

    if not stitched:
        raise SystemExit("没有写出任何帧")
    write_mp4(stitched, out_path, fps or 30, ffmpeg, encoder)
    print(f"已写出: {out_path}  ({len(stitched)} 帧, {fps or 30} fps)")


if __name__ == "__main__":
    main()
