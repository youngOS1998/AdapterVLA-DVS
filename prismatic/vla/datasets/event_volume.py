"""Event volumes and object-center labels for the event network.

The volume is the same square the DVSVLA event encoder reads: two polarities,
eight time bins, and a 32 by 32 image grid over the last 0.4 seconds. The
center label is ``query`` / ``query_ok`` in ``control_switch.jsonl``. The
future label is four displacements: the center at each horizon minus the
center now. With ``future_steps`` 16 the horizons are 4, 8, 12, and 16 frames.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

EVENT_GRID = 32
EVENT_TIME_BINS = 8
EVENT_WINDOW_S = 0.40
DISPLACEMENT_COUNT = 4
GRIPPER_CLOSED = 0.2
GRIPPER_HOLDING = 0.5
BELT_Y_M = 0.0
BELT_Z_M = 0.80
_CV_FROM_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def displacement_horizons(future_steps: int) -> tuple[int, ...]:
    """Four frame gaps ending at ``future_steps``. 16 becomes ``(4, 8, 12, 16)``."""
    last = max(int(future_steps), 1)
    return tuple(max(1, (last * k) // DISPLACEMENT_COUNT) for k in range(1, DISPLACEMENT_COUNT + 1))


def motion_volume(
    events: np.ndarray,
    offset: np.ndarray,
    frame: int,
    height: int,
    width: int,
    window_s: float = EVENT_WINDOW_S,
    time_bins: int = EVENT_TIME_BINS,
    grid: int = EVENT_GRID,
) -> np.ndarray:
    """Events in the recent window, on a square image grid. Shape ``(2, time, grid, grid)``."""
    volume = np.zeros((2, time_bins, grid, grid), dtype=np.float32)
    chosen, t0, t1 = _window_events(events, offset, frame, window_s)
    if len(chosen) == 0 or height <= 0 or width <= 0 or t1 <= t0:
        return volume
    span = max(t1 - t0, 1e-6)
    times = np.clip(((chosen[:, 2] - t0) / span * time_bins).astype(np.int32), 0, time_bins - 1)
    xs = np.clip((chosen[:, 0] / width * grid).astype(np.int32), 0, grid - 1)
    ys = np.clip((chosen[:, 1] / height * grid).astype(np.int32), 0, grid - 1)
    positive = chosen[:, 3] > 0
    for polarity, mask in ((0, ~positive), (1, positive)):
        if np.any(mask):
            np.add.at(volume[polarity], (times[mask], ys[mask], xs[mask]), 1.0)
    residual = np.clip(volume, 0.0, None)
    np.log1p(residual, out=residual)
    return residual


def _window_events(events, offset, frame, window_s):
    empty = np.zeros((0, 4), dtype=np.float32)
    if events is None or len(events) == 0 or len(offset) < 2:
        return empty, 0.0, 0.0
    last = min(int(frame), len(offset) - 2)
    t_end = None
    for index in range(last, max(-1, last - 8), -1):
        chunk = _frame_chunk(events, offset, index)
        if len(chunk):
            t_end = float(chunk[:, 2].max())
            break
    if t_end is None:
        return empty, 0.0, 0.0
    t0 = t_end - float(window_s)
    parts = []
    for index in range(last, -1, -1):
        chunk = _frame_chunk(events, offset, index)
        if len(chunk) == 0:
            continue
        if float(chunk[:, 2].max()) < t0:
            break
        kept = chunk[chunk[:, 2] > t0]
        if len(kept):
            parts.append(kept)
        if float(chunk[:, 2].min()) <= t0:
            break
        if last - index > 16:
            break
    if not parts:
        return empty, t0, t_end
    chosen = parts[0] if len(parts) == 1 else np.concatenate(parts[::-1], axis=0)
    return chosen, t0, t_end


def _frame_chunk(events, offset, frame):
    if frame < 0 or frame + 1 >= len(offset):
        return np.zeros((0, 4), dtype=np.float32)
    return events[int(offset[frame]) : int(offset[frame + 1])]


def object_image_centers(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Normalized ``(x, y)`` of the target, and whether that point is inside the image."""
    path = Path(path)
    clock = _clock_row(path)
    with h5py.File(path, "r") as handle:
        shape = handle["vision/cam_head/shape"][:2]
        height, width = int(shape[0]), int(shape[1])
        ee = handle["state/right_ee_poses"][:, :3].astype(np.float64)
        grip = np.asarray(handle["state/right_ee_joint_states"][:], dtype=np.float64).reshape(ee.shape[0], -1).mean(axis=1)
        intrinsic = handle["vision/cam_head/intrinsic_matrix"][:].astype(np.float64)
        extrinsic = handle["vision/cam_head/extrinsics_matrix"][:].astype(np.float64)
        offset = handle["vision/cam_head/events/offset"][:].astype(np.int64)
        times = handle["vision/cam_head/events/xy_t_p"][:, 2]
    limit = min(len(ee), len(grip), len(intrinsic), len(extrinsic), len(offset) - 1)
    stamps = _frame_times(times, offset, limit)
    closed = np.flatnonzero(grip[:limit] < GRIPPER_CLOSED)
    t_close = int(closed[0]) if closed.size else limit - 1
    opened = np.flatnonzero(grip[t_close + 1 : limit] >= GRIPPER_HOLDING)
    t_release = int(t_close + 1 + opened[0]) if opened.size else limit
    belt_at_close = _belt_point(clock, float(stamps[t_close]))
    offset_xyz = belt_at_close - ee[t_close]
    placed = np.array([float(clock["tray_x"]), float(clock["tray_y"]), BELT_Z_M], dtype=np.float64)
    points = np.zeros((limit, 2), dtype=np.float32)
    valid = np.zeros(limit, dtype=np.bool_)
    for frame in range(limit):
        if frame <= t_close:
            xyz = _belt_point(clock, float(stamps[frame]))
        elif frame < t_release:
            xyz = ee[frame] + offset_xyz
        else:
            xyz = placed
        projected = _project(xyz, intrinsic[frame], extrinsic[frame], width, height)
        if projected is None:
            points[frame] = 0.5
            continue
        points[frame] = projected[0]
        valid[frame] = projected[1]
    return points, valid


def _belt_point(clock: dict, timestamp: float) -> np.ndarray:
    traveled = float(clock["block_speed"]) * (timestamp - float(clock["close_time_s"]))
    return np.array([float(clock["grasp_x"]) + traveled, BELT_Y_M, BELT_Z_M], dtype=np.float64)


def _project(xyz, intrinsic, extrinsic, width, height):
    camera = _CV_FROM_GL @ np.linalg.inv(extrinsic) @ np.array([xyz[0], xyz[1], xyz[2], 1.0])
    if camera[2] <= 1e-6:
        return None
    pix = intrinsic @ camera[:3]
    u = float(pix[0] / camera[2] / max(width, 1))
    v = float(pix[1] / camera[2] / max(height, 1))
    inside = 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0
    clipped = np.array([np.clip(u, 0.0, 1.0), np.clip(v, 0.0, 1.0)], dtype=np.float32)
    return clipped, inside


def _frame_times(times, offset, limit):
    stamps = np.full(limit, np.nan, dtype=np.float64)
    for index in range(limit):
        chunk = times[int(offset[index]) : int(offset[index + 1])]
        if len(chunk):
            stamps[index] = float(chunk.min())
    good = np.isfinite(stamps)
    if not np.any(good):
        return np.arange(limit, dtype=np.float64) * 0.12
    positions = np.arange(limit, dtype=np.float64)
    return np.interp(positions, positions[good], stamps[good])


def _clock_row(path: Path) -> dict:
    clock_path = path.resolve().parent.parent / "clock.jsonl"
    if not clock_path.is_file():
        raise FileNotFoundError(f"找不到目标物轨迹 {clock_path}")
    episode = _episode_id(path)
    chosen = None
    for line in clock_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row["episode"]) == episode:
            chosen = row
    if chosen is None:
        raise KeyError(f"{clock_path} 里没有 episode {episode}")
    return chosen


def _episode_id(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    if not digits:
        raise ValueError(f"轨迹文件名里没有编号: {path.name}")
    return int(digits)


def _as_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, str):
        return value
    item = np.asarray(value).reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode()
    return str(item)


def _as_int(value) -> int:
    return int(np.asarray(value).reshape(-1)[0])


def resolve_hdf5(file_path: str, event_data_root: str = "") -> Path:
    """Use the path stored in RLDS, then the same relative path under ``event_data_root``."""
    recorded = Path(file_path)
    if recorded.is_file():
        return recorded
    root = Path(event_data_root).expanduser() if event_data_root else None
    if root is not None:
        for candidate in (root / recorded.name, root / recorded.parent.name / recorded.name):
            if candidate.is_file():
                return candidate
        matches = list(root.rglob(recorded.name))
        if len(matches) == 1:
            return matches[0]
    raise FileNotFoundError(
        f"事件轨迹不存在: {recorded}。"
        "RLDS 里的 file_path 指向原始 hdf5。确认该文件还在，或用 --event_data_root 指向存放 episode_*.hdf5 的目录。"
    )


def load_switch_labels(path: Path) -> dict[str, dict]:
    """One record per episode. ``file`` is a path suffix such as ``ur3e_wsg/data/episode_0000000.hdf5``."""
    labels: dict[str, dict] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = str(record["file"]).replace("\\", "/")
        labels[key] = record
    return labels


def lookup_label(path: Path, labels: dict[str, dict]) -> dict | None:
    """Longest path suffix wins, so the same episode index in another task does not match."""
    if not labels:
        return None
    episode = str(Path(path)).replace("\\", "/")
    matches = [(len(key), label) for key, label in labels.items() if episode.endswith(key)]
    if not matches:
        return None
    matches.sort(key=lambda item: item[0])
    return matches[-1][1]


def _switch_files(hdf5: Path, event_data_root: str) -> list[Path]:
    """Annotation file beside the episode, or under ``event_data_root`` when RLDS points elsewhere."""
    found: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        if not path.is_file():
            return
        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        found.append(resolved)

    add(hdf5.resolve().parent.parent / "control_switch.jsonl")
    root = Path(event_data_root).expanduser() if event_data_root else None
    if root is not None and root.exists():
        add(root / "control_switch.jsonl")
        add(root / "ur3e_wsg" / "control_switch.jsonl")
        if not found:
            for path in sorted(root.rglob("control_switch.jsonl")):
                add(path)
    return found


def centers_from_switch(path: Path, event_data_root: str, limit: int, cache: dict[Path, dict[str, dict]]) -> tuple[np.ndarray, np.ndarray] | None:
    """Read ``query`` and ``query_ok``. Returns None when this episode has no stored center."""
    for switch in _switch_files(path, event_data_root):
        labels = cache.get(switch)
        if labels is None:
            labels = load_switch_labels(switch)
            cache[switch] = labels
        label = lookup_label(path, labels)
        if label is None or "query" not in label or "query_ok" not in label:
            continue
        queries = np.asarray(label["query"], dtype=np.float32).reshape(-1, 2)
        valid = np.asarray(label["query_ok"], dtype=np.bool_).reshape(-1)
        if len(queries) < limit or len(valid) < limit:
            continue
        return queries[:limit], valid[:limit]
    return None


class _EpisodeEvents:
    def __init__(self, path: Path, event_data_root: str, label_cache: dict[Path, dict[str, dict]]):
        with h5py.File(path, "r") as handle:
            shape = handle["vision/cam_head/shape"][:2]
            self.height = int(shape[0])
            self.width = int(shape[1])
            self.events = handle["vision/cam_head/events/xy_t_p"][:].astype(np.float32)
            self.offset = handle["vision/cam_head/events/offset"][:].astype(np.int64)
        limit = max(len(self.offset) - 1, 0)
        stored = centers_from_switch(path, event_data_root, limit, label_cache)
        if stored is None:
            self.queries, self.query_valid = object_image_centers(path)
            self.center_source = "projection"
        else:
            self.queries, self.query_valid = stored
            self.center_source = "control_switch"
        self.frames = min(len(self.queries), limit)


class EventVolumeIndex:
    """Lazy per-episode cache. One lookup is one training sample."""

    def __init__(self, event_data_root: str = ""):
        self.event_data_root = event_data_root
        self._episodes: dict[str, _EpisodeEvents] = {}
        self._label_cache: dict[Path, dict[str, dict]] = {}

    def sample(self, file_path, frame, future_steps: int) -> dict[str, torch.Tensor]:
        text = _as_text(file_path)
        index = _as_int(frame)
        episode = self._episodes.get(text)
        if episode is None:
            episode = _EpisodeEvents(resolve_hdf5(text, self.event_data_root), self.event_data_root, self._label_cache)
            self._episodes[text] = episode
        blank = {
            "events": torch.zeros(2, EVENT_TIME_BINS, EVENT_GRID, EVENT_GRID),
            "query": torch.zeros(2),
            "query_mask": torch.zeros(()),
            "future_center": torch.zeros(DISPLACEMENT_COUNT, 2),
            "future_center_mask": torch.zeros(DISPLACEMENT_COUNT),
        }
        if index < 0 or index >= episode.frames:
            return blank
        volume = motion_volume(episode.events, episode.offset, index, episode.height, episode.width)
        query = episode.queries[index]
        query_ok = bool(episode.query_valid[index])
        deltas = np.zeros((DISPLACEMENT_COUNT, 2), dtype=np.float32)
        delta_ok = np.zeros(DISPLACEMENT_COUNT, dtype=np.float32)
        if query_ok:
            for slot, horizon in enumerate(displacement_horizons(future_steps)):
                future = index + horizon
                if future < len(episode.queries) and bool(episode.query_valid[future]):
                    deltas[slot] = episode.queries[future] - query
                    delta_ok[slot] = 1.0
        return {
            "events": torch.from_numpy(np.ascontiguousarray(volume)),
            "query": torch.from_numpy(np.ascontiguousarray(query.astype(np.float32))),
            "query_mask": torch.tensor(1.0 if query_ok else 0.0),
            "future_center": torch.from_numpy(np.ascontiguousarray(deltas)),
            "future_center_mask": torch.from_numpy(np.ascontiguousarray(delta_ok)),
        }
