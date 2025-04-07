num_cards=7
bsz_cards=16
time=$(date +%Y%m%d_%H%M%S)
run_id=fake_${time}
mkdir ./${run_id}--image_aug

export WANDB_API_KEY="231c840bf4c83c49cc2241bcce066cb7b75967b2"
export HF_HOME="/aifs4su/mmcode/worldm/.cache/huggingface"

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 /aifs4su/mmcode/videogen/anaconda3/envs/simpler_env/bin/torchrun --standalone --nnodes 1 --nproc-per-node $num_cards scripts/train_vgmvla.py \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix fractal20220817_data \
  --vla.expected_world_size $num_cards \
  --vla.global_batch_size $(expr $bsz_cards \* $num_cards) \
  --vla.per_device_batch_size $bsz_cards \
  --vla.learning_rate 2e-5 \
  --run_root_dir "/aifs4su/mmcode/worldm/videoact/VgmACT" \
  --data_root_dir "/aifs4su/mmcode/worldm/open_x_embodiment/fractal20220817_data/drawer/open_top_drawer" \
  --image_aug True \
  --save_interval 150000 \
  --run_id ${run_id} \
  --repeated_diffusion_steps 8 \
  --future_action_window_size 15 \
  --action_model_type DiT-B \
  --wandb_project "RoboCrafter-CogACT-test" \
  --pretrained_checkpoint "/aifs4su/mmcode/worldm/videoact/VgmACT/DynamiCrafter/checkpoints/ww_training_128_v1.0_rt1/checkpoints/epoch=13-step=9000.ckpt"\
  --wandb_entity 'litwellchi' \
  --is_resume False \
  --vgm_param_mode 'freeze' \
  &>> ./${run_id}--image_aug/train.log &