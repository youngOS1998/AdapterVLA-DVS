#!/bin/bash
# Fine-tune VLA-Adapter on RoboTwin place_color_count_blocks_tray_rand
# (single right-arm Franka, 8-D joint delta: Δq7 + gripper).
# Convert hdf5 first if needed:
#   /root/anaconda3/envs/vla-adapter/bin/python \
#       data_process/convert_byang_lerobot_to_rlds.py --overwrite

set -e
cd /mnt/Projects_origin/VLA-Adapter

current_time=$(date +"%Y%m%d_%H%M%S")
data_name=place_color_count_blocks_tray_rand
mkdir -p logs

# Prefer this repo's prismatic over /mnt/Projects/VLA-Adapter on PYTHONPATH.
export PYTHONPATH="/mnt/Projects_origin/VLA-Adapter${PYTHONPATH:+:${PYTHONPATH}}"

# Activate env if torchrun is not already on PATH.
if ! command -v torchrun >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source /root/anaconda3/etc/profile.d/conda.sh
  conda activate vla-adapter
fi

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
  --num_steps_before_decay 5000 \
  --max_steps 20005 \
  --save_freq 2500 \
  --save_latest_checkpoint_only True \
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
echo "expect:  Using FRANKA_JOINT constants / ACTION_DIM = 8 / PROPRIO_DIM = 8"
