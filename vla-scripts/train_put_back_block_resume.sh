#!/bin/bash
# Continue put_back_block training from the step-40000 checkpoint.
# That directory was saved with save_latest_checkpoint_only, so the files on
# disk are step 40000 (the run itself stopped at 50005 without a final save).
#
# Keep --use_minivlm True, matching the original run. The prompt must contain
# 64 action placeholders; turning minivlm off crashes the first forward.
# finetune.py loads model.safetensors from --resum_vla_path when resume is set.
# Action head and proprio projector are loaded from the same directory.
# Optimizer state is not in the checkpoint, so the LR schedule starts over.

set -e
cd /mnt/Projects_origin/VLA-Adapter

current_time=$(date +"%Y%m%d_%H%M%S")
data_name=put_back_block
resume_step=40000
# log_step = resume_step + new steps, and training stops when log_step hits max_steps.
# 90000 means 50000 additional steps.
max_steps=90000
mkdir -p logs

CKPT="outputs/configs+put_back_block+b64+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter--put_back_block--20260928_202210"

if [[ ! -f "$CKPT/model.safetensors" ]]; then
  echo "missing merged weights: $CKPT/model.safetensors" >&2
  exit 1
fi

ln -sfn action_head--latest_checkpoint.pt "$CKPT/action_head--${resume_step}_checkpoint.pt"
ln -sfn proprio_projector--latest_checkpoint.pt "$CKPT/proprio_projector--${resume_step}_checkpoint.pt"

export PYTHONPATH="/mnt/Projects_origin/VLA-Adapter${PYTHONPATH:+:${PYTHONPATH}}"

if ! command -v torchrun >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source /root/anaconda3/etc/profile.d/conda.sh
  conda activate vla-adapter
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py \
  --config_file_path pretrained_models/configs \
  --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --data_root_dir /mnt/Projects_origin/VLA-Adapter/data/robotwin_rlds \
  --dataset_name "$data_name" \
  --run_root_dir outputs \
  --run_id_override "${data_name}_resume_from_${resume_step}--${current_time}" \
  --resume True \
  --resume_step "$resume_step" \
  --resum_vla_path "$CKPT" \
  --use_minivlm True \
  --use_film False \
  --num_images_in_input 3 \
  --use_proprio True \
  --use_lora True \
  --use_fz False \
  --image_aug True \
  --num_steps_before_decay 50000 \
  --max_steps "$max_steps" \
  --save_freq 5000 \
  --save_latest_checkpoint_only False \
  --merge_lora_during_training True \
  --batch_size 16 \
  --grad_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --lora_rank 64 \
  --use_pro_version True \
  --wandb_entity none \
  --wandb_project "$data_name" \
  > "logs/VLA-Adapter--${data_name}--resume${resume_step}--${current_time}.log" 2>&1 &

echo "started pid $!  log=logs/VLA-Adapter--${data_name}--resume${resume_step}--${current_time}.log"
echo "follow:  tail -f logs/VLA-Adapter--${data_name}--resume${resume_step}--${current_time}.log"
echo "expect:  Using ALOHA constants / ACTION_DIM = 14 / loaded action_head and proprio_projector at step ${resume_step}"
echo "new run: outputs/${data_name}_resume_from_${resume_step}--${current_time}"
