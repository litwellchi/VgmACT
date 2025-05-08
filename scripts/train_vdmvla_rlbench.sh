num_cards=1
bsz_cards=16
time=$(date +%Y%m%d_%H%M%S)
run_id=V3_DiTB_close_box #_${time}
mkdir ./${run_id}--image_aug

export WANDB_API_KEY="231c840bf4c83c49cc2241bcce066cb7b75967b2"
export HF_HOME="/aifs4su/mmcode/worldm/.cache/huggingface"
export TFDS_DATA_DIR="/aifs4su/mmcode/worldm/open_x_embodiment/rlbench/dataset/close_box"


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /aifs4su/mmcode/videogen/anaconda3/envs/simpler_env/bin/torchrun --standalone --nnodes 1 --nproc-per-node $num_cards scripts/train_vgmvla.py \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix custom_finetuning \
  --vla.expected_world_size $num_cards \
  --vla.global_batch_size $(expr $bsz_cards \* $num_cards) \
  --vla.per_device_batch_size $bsz_cards \
  --vla.learning_rate 5e-6 \
  --run_root_dir "/aifs4su/mmcode/worldm/videoact/VgmACT" \
  --data_root_dir ${TFDS_DATA_DIR} \
  --image_aug True \
  --save_interval 20000 \
  --run_id ${run_id} \
  --repeated_diffusion_steps 8 \
  --future_action_window_size 15 \
  --action_model_type DiT-B \
  --wandb_project "vgmact-rlbench10-V3" \
  --pretrained_checkpoint "/aifs4su/mmcode/worldm/RoboCrafter/save_checkpoints/ww_training_128_4frame_v1.0_rt1_4frame/checkpoints/epoch=74-step=600.ckpt"\
  --wandb_entity 'litwellchi' \
  --is_resume False \
  --vgm_param_mode 'lora' \
  --use_future_frame True \
  --full_ckpt "/aifs4su/mmcode/worldm/videoact/VgmACT/V3_rt1_lora_videoloss_20250505_233802--image_aug/checkpoints/step-025000-epoch-32-loss=0.0356.pt"
  # --pretrain_action_model '/aifs4su/mmcode/worldm/videoact/CogACT/CogACT-Base/checkpoints/CogACT-Base.pt' 
  # &>> ./${run_id}--image_aug/train.log &


