#!/bin/bash
# Single-GPU debug of RoboTwin finetune. Same data path as train_robotwin.sh,
# but one process so a debugger can stop inside the training loop.
# GPUs 0-3 are the full training job; this uses a free card (default 4).
#
# Breakpoints worth setting:
#   prismatic/vla/datasets/datasets.py
#     RLDSDataset.__iter__              raw RLDS sample
#     RLDSBatchTransform.__call__       image / prompt / action tokens / proprio
#   prismatic/util/data_utils.py
#     PaddedCollatorForActionPrediction.__call__
#   vla-scripts/finetune.py
#     the `for batch_idx, batch in enumerate(dataloader)` loop
#     run_forward_pass

current_time=$(date +"%Y%m%d_%H%M%S")
data_name=place_color_count_blocks_tray_rand
mkdir -p logs

export PYTHONPATH="/mnt/Projects_origin/VLA-Adapter${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}

PYTHON=${PYTHON:-/root/anaconda3/envs/vla-adapter/bin/python}

"$PYTHON" -m torch.distributed.run --standalone --nnodes 1 --nproc-per-node 1 \
  vla-scripts/finetune.py \
  --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --config_file_path pretrained_models/configs \
  --data_root_dir /mnt/Projects_origin/VLA-Adapter/data/robotwin_rlds \
  --dataset_name "$data_name" \
  --run_root_dir outputs \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --use_lora True \
  --use_fz False \
  --use_minivlm True \
  --image_aug False \
  --num_steps_before_decay 5000 \
  --max_steps 2 \
  --save_freq 100000 \
  --save_latest_checkpoint_only True \
  --merge_lora_during_training False \
  --batch_size 1 \
  --grad_accumulation_steps 1 \
  --learning_rate 2e-4 \
  --lora_rank 64 \
  --use_pro_version True \
  --shuffle_buffer_size 32 \
  --wandb_entity none \
  --wandb_project "${data_name}_debug" \
  --run_id_note "VLA-Adapter--${data_name}--1gpu-debug--${current_time}"
