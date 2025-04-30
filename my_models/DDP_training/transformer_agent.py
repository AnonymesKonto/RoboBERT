import sys
sys.path.append("./my_models/DDP_training")
from calvin_agent.models.calvin_base_model import CalvinBaseModel
import torch
import numpy as np
import cv2 as cv
from my_utils import lang_to_embed, remove_prefix
import sys
import open_clip
from PIL import Image
from ModalityFusioner import ModalityFusioner, LanguageConnecter
from diffusion_modules import ConditionalUnet1D
from torch import nn
import queue
import copy
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

class GPTAgent(CalvinBaseModel):
    def __init__(self, checkpoint, speed_factor):
        super().__init__()
        #print("you are using deepyoloB agent")
        self.checkpoint = checkpoint
        self.actor = None

        pred_horizon = 16
        obs_horizon = 2
        action_horizon = 9
        obs_dim = 768
        action_dim = 7

        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.pred_horizon = pred_horizon
        self.speed_factor = speed_factor

        # create network object
        noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_dim*obs_horizon
        )

        # create vision encoder
        clip_vision_encoder_path: str = "ViT-B-32"
        clip_vision_encoder_pretrained: str = "openai"
        clip_encoder, _, image_processor = open_clip.create_model_and_transforms(
            clip_vision_encoder_path, pretrained=clip_vision_encoder_pretrained
        )
        vision_encoder = clip_encoder.visual
        self.image_processor = image_processor

        modality_fusioner = ModalityFusioner(vision_encoder=vision_encoder, num_history=obs_horizon)

        self.actor = nn.ModuleDict({
                'modality_fusioner': modality_fusioner,
                'noise_pred_net': noise_pred_net,
                'language_connecter': LanguageConnecter()
            })
        
        model_state_dict = remove_prefix(checkpoint['model_state_dict'], isEval=True)
        #print(model_state_dict)
        msg = self.actor.load_state_dict(model_state_dict)
        print(msg)
        self.actor = self.actor.eval()
        #torch.cuda.set_device(0)
        
        self.device = torch.device("cuda", 0)
        torch.cuda.set_device(self.device)
        self.actor.cuda()
        
        self.history = dict()

        # memory for modalities
        self.rgb_static_queue = queue.Queue()
        self.rgb_gripper_queue = queue.Queue()
        self.action_output_num = 0

        self.val_annotations = None

        self.rgb_statics = torch.zeros(1, self.obs_horizon, 3, 224, 224).to(self.device)
        self.rgb_grippers = torch.zeros(1, self.obs_horizon, 3, 224, 224).to(self.device)
        self.action = None

        self.num_diffusion_iters = 10
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule='squaredcos_cap_v2',
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type='epsilon'
        )
        
    def step(self, obs, language):
        rgb_static = self.image_processor(Image.fromarray(obs["rgb_obs"]['rgb_static'])).to(self.device)
        rgb_gripper = self.image_processor(Image.fromarray(obs["rgb_obs"]['rgb_gripper'])).to(self.device)
        
        while self.rgb_static_queue.qsize() < self.obs_horizon:
            self.rgb_static_queue.put(rgb_static)
            self.rgb_gripper_queue.put(rgb_gripper)

        if self.action_output_num == 0:
            #rgb_static_queue = copy.deepcopy(self.rgb_static_queue)
            #rgb_gripper_queue = copy.deepcopy(self.rgb_gripper_queue)

            for i in range(self.obs_horizon):
                temp = self.rgb_static_queue.get()
                self.rgb_statics[0][i] = temp
                self.rgb_static_queue.put(temp)

                temp = self.rgb_gripper_queue.get()
                self.rgb_grippers[0][i] = temp
                self.rgb_gripper_queue.put(temp)

            language_x = self.val_annotations[language]["languages_x"][1].unsqueeze(0).to(self.device)
            language_x = self.actor['language_connecter'](language_x)
            obs_cond = self.actor['modality_fusioner'](
                        self.rgb_statics, self.rgb_grippers, language_x
                    )
            naction = torch.randn((1, self.pred_horizon, 7), device=self.device)

            self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

            for k in self.noise_scheduler.timesteps:
                    # predict noise
                    noise_pred = self.actor['noise_pred_net'](
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond
                    )

                    # inverse diffusion step (remove noise)
                    naction = self.noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample

            self.action = naction[0]
        
        action = self.action[self.action_output_num + self.obs_horizon - 1]
        if action[-1] > 0:
            action[-1] = 1
        else:
            action[-1] = -1
        self.action_output_num = (self.action_output_num + 1) % self.action_horizon
        self.rgb_static_queue.get()
        self.rgb_gripper_queue.get()
        
        #action[0:6] = torch.round(action[0:6] * 20)/20
        action[0:6] *= self.speed_factor

        return action.cpu().numpy()

    def reset(self):
        self.action_output_num = 0
        self.rgb_static_queue.queue.clear()
        self.rgb_gripper_queue.queue.clear()
        
class AgentDebug:
    def __init__(self, actor=None, tokenizer=None):
        pass

    def step(self, obs, language):
        rgb = torch.FloatTensor(obs["rgb_obs"]['rgb_static'])
        action_displacement = np.random.uniform(low=-1, high=1, size=6)
        # And a binary gripper action, -1 for closing and 1 for oppening
        action_gripper = np.random.choice([-1, 1], size=1)
        action = np.concatenate((action_displacement, action_gripper), axis=-1)
        return action
    
    def reset(self):
        pass

class ActorDebug:
    def __init__(self):
        pass

class TokenizerDebug:
    def __init__(self):
        pass
