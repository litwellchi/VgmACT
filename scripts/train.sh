num_cards=4
bsz_cards=16
time=$(date +%Y%m%d_%H%M%S)
run_id=openvla_opentopdrawer_freeze_${time}
mkdir ./${run_id}--image_aug

# config wandb key
export WANDB_API_KEY="231c840bf4c83c49cc2241bcce066cb7b75967b2"

export HF_HOME="/aifs4su/mmcode/worldm/.cache/huggingface"


CUDA_VISIBLE_DEVICES=0,1,2,3 /aifs4su/mmcode/videogen/anaconda3/envs/simpler_env/bin/torchrun --standalone --nnodes 1 --nproc-per-node $num_cards scripts/train.py \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix fractal20220817_data \
  --vla.expected_world_size $num_cards \
  --vla.global_batch_size $(expr $bsz_cards \* $num_cards) \
  --vla.per_device_batch_size $bsz_cards \
  --vla.learning_rate 2e-5 \
  --run_root_dir "/aifs4su/mmcode/worldm/CogACT" \
  --data_root_dir "/aifs4su/mmcode/worldm/open_x_embodiment/fractal20220817_data/drawer/open_top_drawer" \
  --image_aug True \
  --save_interval 1500 \
  --run_id  ${run_id}  \
  --repeated_diffusion_steps 8 \
  --future_action_window_size 15 \
  --action_model_type DiT-S \
  --wandb_project "CogACT-rt1-pretrain-2tasks" \
  --wandb_entity 'litwellchi' \
  --is_resume False \
  --pretrained_checkpoint "/aifs4su/mmcode/videogen/share_ckpts/openvla/openvla-7b-prismatic/checkpoints/model.pt" \
  &>> ./${run_id}--image_aug/train.log &


  # CUDA_VISIBLE_DEVICES=1 /aifs4su/mmcode/videogen/anaconda3/envs/simpler_env/bin/torchrun --standalone --nnodes 1 --nproc-per-node $num_cards scripts/train.py   --vla.type prism-dinosiglip-224px+oxe+diffusion   --vla.data_mix fractal20220817_data   --vla.expected_world_size $num_cards   --vla.global_batch_size $(expr $bsz_cards \* $num_cards)   --vla.per_device_batch_size $bsz_cards   --vla.learning_rate 2e-5   --run_root_dir "/aifs4su/mmcode/worldm/CogACT"   --data_root_dir "/aifs4su/mmcode/worldm/open_x_embodiment/fractal20220817_data/2tasks"   --image_aug True   --save_interval 10000   --run_id 2tasks_$(date +%Y%m%d_%H%M%S)   --repeated_diffusion_steps 8   --future_action_window_size 15   --action_model_type DiT-B   --wandb_project "CogACT-rt1-pretrain-2tasks"   --wandb_entity 'litwellchi'   --is_resume False   --pretrained_checkpoint "/aifs4su/mmcode/videogen/share_ckpts/openvla/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt"