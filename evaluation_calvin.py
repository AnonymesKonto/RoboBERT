from pathlib import Path
from omegaconf import OmegaConf
import hydra
import json
from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    collect_plan,
    count_success,
    create_tsne,
    get_default_model_and_env,
    get_env_state_for_initial_condition,
    get_log_dir,
    join_vis_lang,
    print_and_save,
)
from calvin_env.envs.play_table_env import get_env
from tqdm import tqdm
import torch
#import keyboard
import sys
from preprocessing_val_language import preprocessing_val_languages_clip, preprocessing_val_languages_bert
import numpy as np
import cv2 as cv
import json
import argparse

EP_LEN = 360 #360

NUM_SEQUENCES = 1000

observation_sapce = {'rgb_obs': ['rgb_static', 'rgb_gripper'], 'depth_obs': [], 'state_obs': ['robot_obs'], 'actions': ['rel_actions'], 'language': ['language']}

def make_env(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    env = get_env(val_folder, show_gui=True)

    # insert your own env wrapper
    # env = Wrapper(env)
    return env

def eval_one_epoch_calvin(checkpoint, dataset_path, calvin_conf_path, speed_factor):
    env = make_env(dataset_path)
    model_type = "second"
       
    if model_type == "second":
        from my_models.DDP_training.transformer_agent import GPTAgent
        agent_model = GPTAgent(checkpoint, speed_factor)
        
    evaluate_policy(agent_model, env, calvin_conf_path)

def evaluate_policy(model, env, calvin_conf_path):

    conf_dir = Path(calvin_conf_path)
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)

    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    val_annotations = preprocessing_val_languages_bert(val_annotations)
    model.val_annotations = val_annotations

    eval_sequences = get_sequences(NUM_SEQUENCES,10)

    results = []

    eval_sequences = tqdm(eval_sequences, position=0, leave=True)
    input("press enter to start...")
    for initial_state, eval_sequence in eval_sequences:
        model.reset()
        result = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations)
        results.append(result)

        eval_sequences.set_description(
            " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
        )

    return results

def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence, val_annotations):

    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    
    for subtask_i, subtask in enumerate(eval_sequence):
        #model.reset()
        #print("the language instruction is:",val_annotations[subtask][0])
        success = rollout(env, model, task_checker, subtask, val_annotations)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def rollout(env, model, task_oracle, subtask, val_annotations):

    obs = env.get_obs()
    # get lang annotation for subtask
    
    lang_annotation = subtask
    #lang_annotation = lang_annotation.split('\n')[0]

    #lang_annotation = subtask#.replace("_"," ")
    print("the task category is:",lang_annotation)
    
    start_info = env.get_info()
    
    for step in range(EP_LEN):
        #print(lang_annotation)
        #cv.imshow("static", obs["rgb_obs"]['rgb_static'])
        #cv.waitKey(1)
        action = model.step(obs, lang_annotation)
        #print(action)
        obs, _, _, current_info = env.step(action)

        #if keyboard.is_pressed('q'):
        #    return True        
        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        
        if len(current_task_info) > 0:
            if "move_slider" in lang_annotation:
                for step in range(10):
                    obs, _, _, current_info = env.step(action)
                    action = model.step(obs, lang_annotation)
                    #print(action)
            
            return True

    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='configurating the extraction specifications')
    parser.add_argument('--ckpt_path', type=str, default="path/to/ckpt")
    parser.add_argument('--speed_factor', type=float, default=1.0)
    args = parser.parse_args()

    ckpt_path = args.ckpt_path
    speed_factor = args.speed_factor
    checkpoint = torch.load(ckpt_path)
    
    loaded_config = dict()
    with open("config_path.json", "r") as f:
        loaded_config = json.load(f)

    dataset_path = loaded_config["CALVIN_dataset_path"] + "task_ABCD_D"
    calvin_conf_path = "./calvin_models/conf"

    torch.manual_seed(0)
    with torch.no_grad():
        eval_one_epoch_calvin(checkpoint, dataset_path, calvin_conf_path, speed_factor)
