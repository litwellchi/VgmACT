import os, sys, pathlib
import argparse
import tqdm
import shutil
from termcolor import cprint, colored
import time
import subprocess
from rlbench_env import RLBenchEnv, RLBenchActionMode, RLBenchObservationConfig
from helpers.gymnasium import VideoWrapper
from helpers.common import Logger, save_frames_as_video
from helpers.graphics import EEpose

import numpy as np
import pickle

from vla import load_vgmvla
import torch
from PIL import Image

def recreate_directory(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path, exist_ok=True)

def model_load(args):
        #     self.vla = load_vgmvla(
        #   vgm_base_path,                       # choose from ['CogACT/CogACT-Small', 'CogACT/CogACT-Base', 'CogACT/CogACT-Large'] or the local path
        #   load_for_training=False, 
        #   action_model_type=action_model_type,              # choose from ['DiT-Small', 'DiT-Base', 'DiT-Large'] to match the model weight
        #   future_action_window_size=future_action_window_size,
        #   action_dim=action_dim,
        #   vgm_param_mode=vgm_param_mode,
        #   full_ckpt=saved_model_path
        # )
    model = load_vgmvla(
            args.vgm_base_path,
            load_for_training=False,
            action_model_type=args.action_model_type,
            future_action_window_size=int(args.model_action_steps),
            action_dim=args.action_dim,
            vgm_param_mode=args.vgm_param_mode,
            full_ckpt=args.saved_model_path,
            hf_token=args.hf_token
            )
    model.to(f'cuda:{args.cuda}').eval()
    return model

def model_predict(model, image, prompt):
    actions, _ = model.predict_action(
            image,
            prompt,
            unnorm_key='rlbench',
            cfg_scale = 1.5, 
            use_ddim = True,
            num_ddim_steps = 10,
            )
    return actions

def cal_cos(a,b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    cosine_similarity = dot_product / (norm_a * norm_b + 1e-7)
    return cosine_similarity

def main(args):
    # Report the arguments
    Logger.log_info(f'Running {colored(__file__, "red")} with arguments:')
    Logger.log_info(f'task name: {args.task_name}')
    Logger.log_info(f'VGM base setting: {args.vgm_base_path}')
    Logger.log_info(f'Evaluating checkpoint: {args.saved_model_path}')
    Logger.log_info(f'number of episodes: {args.num_episodes}')
    Logger.log_info(f'result directory: {args.result_dir}')
    Logger.log_info(f'replay data directory: {args.replay_data_dir}')
    Logger.log_info(f'exp name: {args.exp_name}')
    Logger.log_info(f'actions steps: {args.model_action_steps}')
    Logger.log_info(f'replay or predict: {args.replay_or_predict}')
    Logger.log_info(f'max steps: {args.max_steps}')
    Logger.log_info(f'cuda used: {args.cuda}')
    # cprint('-' * os.get_terminal_size().columns, 'cyan')
    
    action_mode = RLBenchActionMode.eepose_then_gripper_action_mode(absolute=True)
    obs_config = RLBenchObservationConfig.single_view_config(camera_name='front', image_size=(224, 224))
    env = RLBenchEnv(
        task_name=args.task_name,
        action_mode=action_mode,
        obs_config=obs_config,
        point_cloud_camera_names=['front'],
        cinematic_record_enabled=True,
        headless=True
    )
    env = VideoWrapper(env)
    # raise f"{args.task_name} has been created."
    if args.replay_or_predict == 'predict':
        args.result_dir = os.path.join(args.result_dir, 'predict_results')
    elif args.replay_or_predict == 'replay':
        args.result_dir = os.path.join(args.result_dir, 'replay_results')
    
    if args.exp_name is None:
        args.exp_name = args.task_name

    video_dir = os.path.join(
        args.result_dir, args.task_name, args.exp_name, "videos"
    )
    recreate_directory(video_dir)
    
    success_num = 0

    # #----------- for model predict
    if args.replay_or_predict == 'predict':
        print('args : ',args)
        model = model_load(args)
        episode_length = args.max_steps

    t1 = time.time()
    pre_time = 0
    step_time = 0
    cnt = 0
    for i in range(args.num_episodes):
        Logger.log_info(f'episode: {colored(i, "red")}, steps: {colored(episode_length, "red")}')

        try:
            Logger.log_info("Loading enironment ...")
            obs_dict = env.reset()
            terminated = False
            rewards = 0
            success = False
        except:
            Logger.log_info("Error: Loading enironment fail.")
        images=[]
        for j in range(episode_length):
            # # #----------- for model predict
            if args.replay_or_predict == 'predict':
                Logger.log_info("Prediction action ...")
                image = obs_dict['image']
                image = Image.fromarray(image)
                images.append(image)
                robot_state = obs_dict['robot_state']
                prompt = env.text
                t2 = time.time()
                action = model_predict(model, image, prompt)[0]

                pre_time += ((time.time()-t2)/60)

                action[:3] += robot_state[7:10] # TODO 是不是有问题
                gripper_open = action[-1]
                action = EEpose.pose_6DoF_to_7DoF(action[:-1])
                action = np.append(action, gripper_open)
                print(j, "  :", action)


            t3 = time.time()

            try:
                Logger.log_info("Running env.step(action)...")
                obs_dict, reward, terminated, truncated, info = env.step(action)

                step_time += ((time.time()-t3)/60)

                cnt += 1
                
                rewards += reward
                success = success or bool(reward)
            except:
                Logger.log_info("Error running env.step(action). Note as fail.")
            if terminated or truncated or success:
                break
        
        if success:
            success_num += 1

        image_dir = os.path.join(
            args.result_dir, args.task_name, args.exp_name, "images", f"episode{i}"
        ) 
        env.tr.record_end(env.unwrapped.env.rlbench_task_env._scene)
        video_save_path=os.path.join(video_dir, f'episode{i}_video_cinematic_{success}.mp4')
        env.tr.save(video_save_path, 'cinematic video')
        save_frames_as_video(images,os.path.join(video_dir, f'episode{i}_video_observation_{success}.mp4'))
        Logger.log_info(f'video saved to {video_dir}')
        Logger.log_info(f'episode{i}_{success}')
        Logger.print_seperator()

        try:
            subprocess.Popen(["ffmpeg", "-loglevel", "quiet", "-i",  video_save_path , "-vcodec", "libx264", "-acodec", "aac", os.path.join(video_dir, f'episode{i}_video_cinematic_{success}_h264.mp4')])
            subprocess.Popen(["ffmpeg", "-loglevel", "quiet", "-i", os.path.join(video_dir, f'episode{i}_video_observation_{success}.mp4'), "-vcodec", "libx264", "-acodec", "aac", os.path.join(video_dir, f'episode{i}_video_observation_{success}_h264.mp4')])
        except Exception as e:
            print(f"Error starting ffmpeg processes: {e}")
            
    t4 = time.time()

    Logger.log_ok(f'Finished. {args.task_name} * {args.num_episodes}. Success rate {success_num/args.num_episodes*100}%')
    with open(os.path.join(args.result_dir, args.task_name, f'{args.exp_name}_success_rate.txt'), "w", encoding="utf-8") as file:
        file.write(f'Finished. {args.task_name} * {args.num_episodes}. Success rate {success_num/args.num_episodes*100}%, All_episodes_time: {(t4-t1)/60},  Average model inference time: {pre_time/cnt} ,  Average env step time: {step_time/cnt}, Cnt: {cnt}')

    Logger.log_ok(f'All_episodes_time: {(t4-t1)/60},  Average model inference time: {pre_time/cnt} ,  Average env step time: {step_time/cnt}, Cnt: {cnt}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay-data-dir', type=str, default='/home/cx/ch_collect_keypoints_rlbench/for_rlds')
    parser.add_argument('--task-name', type=str, default='close_box')
    parser.add_argument('--replay-or-predict', type=str, default='predict')
    parser.add_argument('--exp-name', type=str, default='exp')
    parser.add_argument('--num-episodes', type=int, default=3)
    parser.add_argument('--model-action-steps', type=str, default='15')
    parser.add_argument('--result-dir', type=str, default='/home/cx/ch_collect_keypoints_rlbench')
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--max-steps', type=int, default=10)
    parser.add_argument('--hf-token', type=str, default='')
    parser.add_argument('--action_dim', type=int, default=7)
    parser.add_argument('--vgm_param_mode', type=str, default='freeze')
    parser.add_argument('--vgm_base_path', type=str, default='/aifs4su/mmcode/worldm/RoboCrafter/save_checkpoints/ww_training_128_4frame_v1.0_rt1_4frame/checkpoints/epoch=74-step=600.ckpt')
    parser.add_argument('--action_model_type', type=str, default='DiT-S')
    parser.add_argument('--saved_model_path', type=str, default='/aifs4su/mmcode/worldm/videoact/VgmACT/V2_DiTS_freeze_128vgm_rlbench10_20250407_164332--image_aug/checkpoints/step-010000-epoch-294-loss=0.0044.pt')
    
    main(parser.parse_args())