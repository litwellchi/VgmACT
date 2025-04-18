# shader_dir=rt means that we turn on ray-tracing rendering; this is quite crucial for the open / close drawer task as policies often rely on shadows to infer depth
# Test setting for Version1.0 of VgmACT



gpu_id=0
export WANDB_API_KEY="231c840bf4c83c49cc2241bcce066cb7b75967b2"
export HF_HOME="/aifs4su/mmcode/worldm/.cache/huggingface"


declare -a ckpt_paths=(
"/aifs4su/mmcode/worldm/videoact/VgmACT/V22_DiTS_lora_128vgm_vidloss_rlbench10_20250409_224329--image_aug/checkpoints/step-030000-epoch-882-loss=0.0049.pt"
"/aifs4su/mmcode/worldm/videoact/VgmACT/V22_DiTS_lora_128vgm_vidloss_rlbench10_20250409_224329--image_aug/checkpoints/step-020000-epoch-588-loss=0.0076.pt"
"/aifs4su/mmcode/worldm/videoact/VgmACT/V22_DiTS_lora_128vgm_vidloss_rlbench10_20250409_224329--image_aug/checkpoints/step-010000-epoch-294-loss=0.0054.pt"
    # V1.0
# "/aifs4su/mmcode/worldm/videoact/VgmACT/DiTS_freeze_128vgm_rlbench10_20250330_113357--image_aug/checkpoints/step-040000-epoch-1000-loss=0.0023.pt"
# "/aifs4su/mmcode/worldm/videoact/VgmACT/DiTS_freeze_128vgm_rlbench10_20250330_113357--image_aug/checkpoints/step-020000-epoch-500-loss=0.0096.pt"
# "/aifs4su/mmcode/worldm/videoact/VgmACT/DiTS_freeze_128vgm_rlbench10_20250330_113357--image_aug/checkpoints/step-004000-epoch-100-loss=0.0198.pt"
)


exp_name_head="V22_DiTS_lora_128vgm_rlbench10"


tasks=('close_fridge' 'put_rubbish_in_bin' 'sweep_to_dustpan' 'phone_on_base' 'change_clock' 'take_umbrella_out_of_umbrella_stand' 'take_frame_off_hanger' 'close_box' 'close_laptop_lid' 'toilet_seat_down')
task_device_mapping=("0" "0" "0" "0" "0" "0" "0" "0" "0" "0")
for model in "${ckpt_paths[@]}"; do
    ckpt_filename=$(basename "$model")
    exp_name="${exp_name_head}_${ckpt_filename}"
    echo "Running experiment: ${exp_name}"
    job_count=0
    for i in ${!tasks[@]}; do
        task="${tasks[$i]}"
        device="${task_device_mapping[$i]}"
        echo "Launching task: $task on device $device"
        xvfb-run -a python replay_test_rlbench_ch.py \
        --replay-data-dir /home/cx/ch_collect_keypoints_rlbench/for_rlds \
        --exp-name ${exp_name} \
        --saved_model_path ${model} \
        --vgm_param_mode 'lora' \
        --vgm_base_path '/aifs4su/mmcode/worldm/RoboCrafter/save_checkpoints/ww_training_128_v1.0_rt1/checkpoints/epoch=13-step=9000.ckpt' \
        --result-dir './' \
        --replay-data-dir './' \
        --task-name ${task} \
        --max-steps 15 \
        --num-episodes 25 \
        --cuda ${device} \
        --replay-or-predict predict \
        --action_model_type 'DiT-S' \
        --hf-token '' &  # 注意这里的 & 表示后台运行

        ((job_count++))

        # 可选：限制并发数，例如最多同时运行 4 个任务
        max_jobs=4
        if (( job_count % max_jobs == 0 )); then
            wait  # 等待当前批次任务完成
        fi
        # break
    done
    # break
        # Wait for remaining background jobs for current ckpt_path
    wait
done