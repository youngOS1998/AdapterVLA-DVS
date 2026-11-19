"""在当前进程里调试事件融合训练，断点打在真实训练代码上。

不要用 vla-scripts/train_grasp_conveyor_rand_speed_2_event.sh 调试。那个脚本会
把训练丢到后台，并且用 torchrun 换掉当前进程，调试器跟着断掉。

数据怎么走（按这个顺序下断点）：

1. prismatic/vla/datasets/datasets.py  RLDSBatchTransform.__call__
   一帧 RGB、动作块、本体状态。事件融合打开时再读事件体和物体中心。
2. prismatic/vla/datasets/event_volume.py  EventVolumeIndex.sample
   hdf5 里的事件收成 (2, 8, 32, 32)，中心来自 control_switch.jsonl。
3. vla-scripts/finetune.py  run_forward_pass
   损失是动作块 L1、当前中心 L1、未来中心 L1。
4. prismatic/models/action_heads.py  predict_action / _encode_events
   事件 token 进 Pro 动作桥，中心头给出当前中心和未来中心。

用法：
  在 Cursor 里选 “VLA-Adapter: Debug Event Train (1 GPU)”，F5。
  只看数据、不加载 VLM：选 “VLA-Adapter: Debug Event Data Only”。
  命令行：
    /root/anaconda3/envs/vla-adapter/bin/python debug_event_train.py
    /root/anaconda3/envs/vla-adapter/bin/python debug_event_train.py --data-only
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _config_path() -> Path:
    if "--config" in sys.argv:
        return Path(sys.argv[sys.argv.index("--config") + 1]).resolve()
    return _ROOT / "train_event_debug.yaml"


def _load_cfg(path: Path) -> dict:
    with path.open() as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise SystemExit(f"调试配置不是字典：{path}")
    return cfg


def _prepare_process(cfg: dict) -> None:
    """单进程、单卡。LOCAL_RANK 让 accelerate 在当前进程里建好通信组，这样 DDP 和 barrier 能用，调试器不用跟子进程。"""
    gpus = cfg.get("gpus", [4])
    if not isinstance(gpus, list) or len(gpus) != 1:
        raise SystemExit(f"调试只能用一张卡，当前 gpus={gpus}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[0])
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29617")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _argv_from_cfg(cfg: dict) -> list[str]:
    argv = [str(_ROOT / "vla-scripts" / "finetune.py")]
    for key, value in cfg.items():
        if key == "gpus" or value is None:
            continue
        if isinstance(value, bool):
            argv.extend([f"--{key}", "True" if value else "False"])
        else:
            argv.extend([f"--{key}", str(value)])
    return argv


def _describe(name: str, batch: dict) -> None:
    print(f"\n[debug] ===== {name} =====", flush=True)
    for key, value in batch.items():
        shape = getattr(value, "shape", None)
        if shape is not None:
            print(f"  {key}: {tuple(shape)} {getattr(value, 'dtype', '')}", flush=True)
        elif isinstance(value, (list, tuple)):
            preview = repr(value[0]) if value else ""
            if len(preview) > 120:
                preview = preview[:120] + "..."
            print(f"  {key}: list[{len(value)}] {preview}", flush=True)
        else:
            text = repr(value)
            if len(text) > 160:
                text = text[:160] + "..."
            print(f"  {key}: {type(value).__name__} {text}", flush=True)


def run_data_only(cfg: dict) -> None:
    """走到第一个 batch 就停，不加载 VLM。"""
    import json

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoConfig, AutoImageProcessor, AutoProcessor

    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder

    config_dir = Path(cfg["config_file_path"])
    if not config_dir.is_absolute():
        config_dir = _ROOT / config_dir
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    processor = AutoProcessor.from_pretrained(config_dir, trust_remote_code=True)
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    with (config_dir / "config.json").open() as handle:
        image_sizes = json.load(handle)["image_sizes"]

    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=int(cfg["num_images_in_input"]) > 1,
        use_proprio=bool(cfg["use_proprio"]),
        use_minivlm=bool(cfg["use_minivlm"]),
        use_event_fusion=bool(cfg["use_event_fusion"]),
        future_center_steps=int(cfg["future_center_steps"]),
        event_data_root=str(cfg["event_data_root"]),
    )
    dataset = RLDSDataset(
        cfg["data_root_dir"],
        cfg["dataset_name"],
        batch_transform,
        resize_resolution=tuple(image_sizes),
        shuffle_buffer_size=int(cfg["shuffle_buffer_size"]),
        future_action_window_size=int(cfg["future_action_window_size"]),
        image_aug=bool(cfg["image_aug"]),
    )
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )
    batch = next(iter(loader))
    _describe("batch", batch)
    events = batch.get("events")
    if events is not None:
        print(
            f"[debug] events sum {events.sum().item():.1f}  "
            f"query {batch['query'].detach().cpu().tolist()}  "
            f"query_ok {batch['query_mask'].detach().cpu().tolist()}  "
            f"future {batch['future_center'].detach().cpu().tolist()}",
            flush=True,
        )
    del torch
    print("[debug] data only done", flush=True)


def main() -> None:
    data_only = "--data-only" in sys.argv
    cfg_path = _config_path()
    cfg = _load_cfg(cfg_path)
    _prepare_process(cfg)
    sys.argv = _argv_from_cfg(cfg)
    print(f"[debug] config {cfg_path}", flush=True)
    print(f"[debug] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}  single process", flush=True)
    if data_only:
        run_data_only(cfg)
        return

    import importlib.util

    finetune_path = _ROOT / "vla-scripts" / "finetune.py"
    spec = importlib.util.spec_from_file_location("finetune_debug", finetune_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.finetune()


if __name__ == "__main__":
    main()
