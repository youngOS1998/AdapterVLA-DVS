#!/bin/bash
# Fine-tune VLA-Adapter on RoboTwin put_back_block
# (Aloha AgileX, both arms, 14-D absolute joints: left q6+grip + right q6+grip).
# Convert hdf5 first if needed:
#   /root/anaconda3/envs/vla-adapter/bin/python \
#       data_process/convert_byang_lerobot_to_rlds.py \
#       --source_dir /mnt/Projects_origin/VLA-Adapter/data/put_back_block \
#       --dataset_name put_back_block --overwrite

set -e
cd /mnt/Projects_origin/VLA-Adapter

current_time=$(date +"%Y%m%d_%H%M%S")
data_name=put_back_block
mkdir -p logs

# Prefer this repo's prismatic over /mnt/Projects/VLA-Adapter on PYTHONPATH.
export PYTHONPATH="/mnt/Projects_origin/VLA-Adapter${PYTHONPATH:+:${PYTHONPATH}}"

# Activate env if torchrun is not already on PATH.
if ! command -v torchrun >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source /root/anaconda3/etc/profile.d/conda.sh
  conda activate vla-adapter
fi

# 3 cameras: head + left wrist + right wrist. 50 demos, so the step budget
# is shorter than the Franka script.
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py \
  --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --config_file_path pretrained_models/configs \
  --data_root_dir /mnt/Projects_origin/VLA-Adapter/data/robotwin_rlds \
  --dataset_name "$data_name" \
  --run_root_dir outputs \
  --use_film False \
  --num_images_in_input 3 \
  --use_proprio True \
  --use_lora True \
  --use_fz False \
  --use_minivlm True \
  --image_aug True \
  --num_steps_before_decay 50000 \
  --max_steps 50005 \
  --save_freq 20000 \
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
echo "expect:  Using ALOHA constants / ACTION_DIM = 14 / PROPRIO_DIM = 14"
