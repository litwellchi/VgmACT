num_cards=8
bsz_cards=16
time=$(date +%Y%m%d_%H%M%S)
run_id=V3_rt1_lora_videoloss_${time}
mkdir ./${run_id}--image_aug

export WANDB_API_KEY="231c840bf4c83c49cc2241bcce066cb7b75967b2"
export HF_HOME="/aifs4su/mmcode/worldm/.cache/huggingface"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /aifs4su/mmcode/videogen/anaconda3/envs/simpler_env/bin/torchrun --standalone --nnodes 1 --nproc-per-node $num_cards scripts/train_vgmvla.py \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix fractal20220817_data \
  --vla.expected_world_size $num_cards \
  --vla.global_batch_size $(expr $bsz_cards \* $num_cards) \
  --vla.per_device_batch_size $bsz_cards \
  --vla.learning_rate 1e-5 \
  --run_root_dir "/aifs4su/mmcode/worldm/videoact/VgmACT" \
  --data_root_dir "/aifs4su/mmcode/worldm/open_x_embodiment/fractal20220817_data/concat_future4_gap4/" \
  --image_aug True \
  --save_interval 50000 \
  --run_id ${run_id} \
  --repeated_diffusion_steps 8 \
  --future_action_window_size 15 \
  --action_model_type DiT-B \
  --wandb_project "vgmact-rt1" \
  --pretrained_checkpoint "/aifs4su/mmcode/worldm/RoboCrafter/save_checkpoints/ww_training_128_4frame_v1.0_rt1_real_4frame/checkpoints/trainstep_checkpoints/epoch=53-step=12000.ckpt"\
  --wandb_entity 'litwellchi' \
  --is_resume False \
  --vgm_param_mode 'lora' \
  --use_future_frame True \
  --full_ckpt "/aifs4su/mmcode/worldm/videoact/VgmACT/V3_rt1_lora20250505_185434--image_aug/checkpoints/step-010000-epoch-344-loss=0.0073.pt"
