num_cards=4
bsz_cards=32
$bsz_cards*$num_cards
time=$(date +%Y%m%d_%H%M%S)
mkdir ./${time}--image_aug
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nnodes 1 --nproc-per-node $num_cards scripts/train.py \
  --pretrained_checkpoint "CogACT/CogACT-Base" \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix fractal20220817_data \
  --vla.expected_world_size $num_cards \
  --vla.global_batch_size $(expr $bsz_cards \* $num_cards) \
  --vla.per_device_batch_size $bsz_cards \
  --vla.learning_rate 2e-5 \
  --run_root_dir "/aifs4su/mmcode/worldm/CogACT" \
  --data_root_dir "/aifs4su/mmcode/worldm/open_x_embodiment/oxe" \
  --image_aug True \
  --save_interval 10 \
  --run_id ${time} \
  --repeated_diffusion_steps 8 \
  --future_action_window_size 15 \
  --action_model_type DiT-B \
  --wandb_project "RoboCrafter-CogACT-test" \
  --wandb_entity 'litwellchi' \
  --is_resume False \
&>> ./${time}--image_aug/train.log &