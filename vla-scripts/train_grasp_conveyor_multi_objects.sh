#!/bin/bash
# Fine-tune VLA-Adapter on grasp_conveyor_multi_objects.
# Same setup as grasp_conveyor_block: single UR3e stored as a duplicated 14-D
# vector. The dataset transform keeps the right arm (q6 + gripper) and one wrist.
#
#   bash vla-scripts/train_grasp_conveyor_multi_objects.sh

set -e
cd /mnt/Projects_origin/VLA-Adapter

current_time=$(date +"%Y%m%d_%H%M%S")
data_name=grasp_conveyor_multi_objects
mkdir -p logs

# Prefer this repo's prismatic over /mnt/Projects/VLA-Adapter on PYTHONPATH.
export PYTHONPATH="/mnt/Projects_origin/VLA-Adapter${PYTHONPATH:+:${PYTHONPATH}}"

# Activate env if torchrun is not already on PATH.
if ! command -v torchrun >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source /root/anaconda3/etc/profile.d/conda.sh
  conda activate vla-adapter
fi

# 2 cameras: head + right wrist. 100 demos.
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py \
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
  --image_aug True \
  --num_steps_before_decay 20000 \
  --max_steps 20005 \
  --save_freq 2500 \
  --save_latest_checkpoint_only False \
  --merge_lora_during_training True \
  --batch_size 16 \
  --grad_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --lora_rank 64 \
  --use_pro_version True \
  --wandb_entity none \
  --wandb_project "$data_name" \
  --run_id_note "VLA-Adapter--${data_name}--${current_time}" \
  > "logs/VLA-Adapter--${data_name}--${current_time}.log" 2>&1 &

echo "started pid $!  log=logs/VLA-Adapter--${data_name}--${current_time}.log"
echo "follow:  tail -f logs/VLA-Adapter--${data_name}--${current_time}.log"
echo "expect:  Using UR3E_JOINT constants / ACTION_DIM = 7 / PROPRIO_DIM = 7"
