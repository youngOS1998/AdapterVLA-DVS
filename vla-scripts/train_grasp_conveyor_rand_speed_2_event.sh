#!/bin/bash
# Fine-tune VLA-Adapter with the event network on grasp_conveyor_rand_speed_2.
# The RGB path is unchanged. Each sample also reads the head-camera event
# volume from the hdf5 path stored in the RLDS episode, and the Pro action
# bridge cross-attends to those tokens.
#
#   bash vla-scripts/train_grasp_conveyor_rand_speed_2_event.sh

set -e
cd /mnt/Projects_origin/VLA-Adapter-DVS

current_time=$(date +"%Y%m%d_%H%M%S")
data_name=grasp_conveyor_rand_speed_2
mkdir -p logs

export PYTHONPATH="/mnt/Projects_origin/VLA-Adapter-DVS${PYTHONPATH:+:${PYTHONPATH}}"

if ! command -v torchrun >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source /root/anaconda3/etc/profile.d/conda.sh
  conda activate vla-adapter
fi

# 2 cameras: head + right wrist. Event fusion reads vision/cam_head/events.
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py \
  --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --config_file_path pretrained_models/configs \
  --data_root_dir /mnt/Projects_origin/VLA-Adapter/data/robotwin_rlds \
  --dataset_name "$data_name" \
  --run_root_dir outputs \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --use_event_fusion True \
  --event_data_root /mnt/Projects_origin/VLA-Adapter-DVS/data/grasp_conveyor_rand_speed_2 \
  --future_center_steps 16 \
  --future_center_weight 1 \
  --use_lora True \
  --use_fz False \
  --use_minivlm True \
  --image_aug True \
  --num_steps_before_decay 40000 \
  --max_steps 40005 \
  --save_freq 5000 \
  --save_latest_checkpoint_only False \
  --merge_lora_during_training True \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --lr_warmup_steps 2000 \
  --max_grad_norm 1.0 \
  --lora_rank 64 \
  --use_pro_version True \
  --wandb_entity none \
  --wandb_project "$data_name" \
  --run_id_note "VLA-Adapter-Event--${data_name}--${current_time}" \
  > "logs/VLA-Adapter-Event--${data_name}--${current_time}.log" 2>&1 &

echo "started pid $!  log=logs/VLA-Adapter-Event--${data_name}--${current_time}.log"
echo "follow:  tail -f logs/VLA-Adapter-Event--${data_name}--${current_time}.log"
echo "expect:  Using UR3E_JOINT constants / event fusion / ACTION_DIM = 7"
