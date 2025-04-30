import random
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import configargparse
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer
import cv2 as cv
import random
import torchvision.transforms as transforms
from data_enhence import AddPepperNoise, AddResizeCrop, AddRotation, EnhenceSequence, AddAffine, QuadrantShuffle
from data_enhence import MySequenceAffine, MySequenceColorJitter, MySequenceRanResizedCrop, KeepCriticalColor
import open_clip
import copy
import pickle

class PretrainCalvinDataset(Dataset):
    """Naive implementation of dataset to store
    calvin debug dataset, may be changed to WDS for the full dataset
    """

    def __init__(self, dataset_path, image_processor, obs_horizon, pred_horizon, is_train=True, test=False) -> None:
        super().__init__()
        print("using all the unlabelled data")
        self.image_processor = image_processor
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_horizon = 3
        self.test = test

        self.language_tokens = 8
        self.language_dim = 768
        
        tag = "training" if is_train else "validation"
        self.file_prefix = f"{dataset_path}/{tag}"

        self.samples_lookup = []
        ep_start_end_ids = np.load(self.file_prefix + "/ep_start_end_ids.npy")
        print(ep_start_end_ids)

        self.dataset_wo_image = None
        if not test:
            with open('dataset_wo_image_ABC_D.pkl', 'rb') as f:
                self.dataset_wo_image = pickle.load(f)

        for start_end_pair in ep_start_end_ids:
            start_idx, end_idx = start_end_pair[0], start_end_pair[1]
            for pred_start_idx in range(start_idx, end_idx - pred_horizon + 2):
                    pred_end_idx = min(end_idx, pred_start_idx + pred_horizon - 1)
                    obs_start_idx = pred_start_idx
                    obs_end_idx = obs_start_idx + obs_horizon - 1
                    sample_info = {
                                    "task_id" : -1, 
                                    "pred_start_idx": pred_start_idx, 
                                    "pred_end_idx": pred_end_idx,
                                    "obs_start_idx": obs_start_idx,
                                    "obs_end_idx": obs_end_idx
                                   }
                    self.samples_lookup.append(sample_info)

        self.vs_enhence = transforms.Compose(
            [
                AddPepperNoise(0.95, p=0.8),
            ]
        )

        self.myStrongCJ = MySequenceColorJitter(brightness=(1.2, 1.6), contrast=0.0, saturation=(0.5,1.1), hue=0.3)
        self.colorKeeper = KeepCriticalColor()

    def __len__(self):
        return len(self.samples_lookup)

    def __getitem__(self, index):
    	# get the language related info
        sample_info = self.samples_lookup[index]
        task = "no label"

        obs_start_idx = sample_info["obs_start_idx"]
        obs_end_idx = sample_info["obs_end_idx"]
        pred_start_idx = sample_info["pred_start_idx"]
        pred_end_idx = sample_info["pred_end_idx"]

        # get the observation and action in this timespan, the time before the sequence starting will be filled with 0
        vs_height = 224; vs_width = 224; vs_channel = 3
        vg_height = 224; vg_width = 224; vg_channel = 3
        
        prop_dim = 15
        action_dim = 7

        rgb_static = torch.zeros(self.obs_horizon, vs_channel, vs_height, vs_width)
        rgb_gripper = torch.zeros(self.obs_horizon, vg_channel, vg_height, vg_width)        
        robot_obs = torch.zeros(self.obs_horizon, prop_dim)
        rel_actions = torch.zeros(self.pred_horizon, action_dim)
        padding_mask = torch.ones(self.pred_horizon)
        language_x = torch.rand(self.language_tokens, self.language_dim)

        img_frames = [
            np.load(f"{self.file_prefix}/episode_{frame_idx:07d}.npz")
            for frame_idx in range(pred_start_idx, pred_start_idx + self.obs_horizon)
        ]
        
        self.myStrongCJ.set_params()
        #self.myWeakCJ.set_params()
        #self.mySeqAff_s.set_params()

        for i, frame_idx in enumerate(range(pred_start_idx, pred_end_idx+1)):
            if i < self.obs_horizon:
                rgb_static_i = Image.fromarray(img_frames[i]["rgb_static"])
                rgb_gripper_i = Image.fromarray(img_frames[i]["rgb_gripper"])
                
                ori_rgb_static_i = copy.deepcopy(rgb_static_i)
                ori_rgb_gripper_i = copy.deepcopy(rgb_gripper_i)

                rgb_static_i = self.myStrongCJ.ex(rgb_static_i)
                rgb_gripper_i = self.myStrongCJ.ex(rgb_gripper_i)

                #rgb_static_i = self.mySeqAff_s.ex(rgb_static_i)
                rgb_static_i = self.colorKeeper.recover_critical_color("all", ori_rgb_static_i, rgb_static_i)
                rgb_gripper_i = self.colorKeeper.recover_critical_color("all", ori_rgb_gripper_i, rgb_gripper_i)

                rgb_static_i = self.vs_enhence(rgb_static_i)
                rgb_gripper_i = self.vs_enhence(rgb_gripper_i)

                if self.test:
                    np_s = np.array(rgb_static_i)
                    np_v = np.array(rgb_gripper_i)

                    np_s = cv.cvtColor(np_s, cv.COLOR_RGB2BGR)
                    np_v = cv.cvtColor(np_v, cv.COLOR_RGB2BGR)

                    cv.imshow("np_s",np_s)
                    cv.imshow("np_v",np_v)

                    ori_np_s = np.array(ori_rgb_static_i)
                    ori_np_v = np.array(ori_rgb_static_i)

                    ori_np_s = cv.cvtColor(ori_np_s, cv.COLOR_RGB2BGR)
                    ori_np_v = cv.cvtColor(ori_np_v, cv.COLOR_RGB2BGR)

                    cv.imshow("ori_np_s",ori_np_s)
                    cv.imshow("ori_np_v",ori_np_v)
                    cv.waitKey(0)
            
                rgb_static[i] = self.image_processor(rgb_static_i)
                rgb_gripper[i] = self.image_processor(rgb_gripper_i)
            
            padding_mask[i] = 0
            if not self.test:
                rel_actions[i] = torch.from_numpy(self.dataset_wo_image[frame_idx]["rel_actions"][0:7])

        observe_space = {"rgb_static": rgb_static, "robot_obs": robot_obs, "rgb_gripper":rgb_gripper}
        action_space = {"rel_actions": rel_actions}

        return observe_space, task, language_x, padding_mask, action_space            


if __name__ == "__main__":
    print("test")

    dataset_path = "/home/wibot/Data/SC/task_ABC_D"
    device = torch.device("cuda:0") 
    num_history = 30

    # define the training style
    
    clip_vision_encoder_path: str = "ViT-B-32"
    clip_vision_encoder_pretrained: str = "openai"
    clip_encoder, _, image_processor = open_clip.create_model_and_transforms(
        clip_vision_encoder_path, pretrained=clip_vision_encoder_pretrained
    )
    calvin_dataset = PretrainCalvinDataset(
        dataset_path=Path(dataset_path),
        image_processor=image_processor,
        obs_horizon=140, 
        pred_horizon=160,
        test=True
    )
    print(len(calvin_dataset))

    calvin_dataset.__getitem__(1401510) #1401510
