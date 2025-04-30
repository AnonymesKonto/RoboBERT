import torch
from torch import nn
from positional_encoding import PositionalEncoding
from torch.nn.modules.transformer import _generate_square_subsequent_mask
#from observation_encoder import make_CNN_vision_encoder
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import Tensor
from typing import Optional, Any, Union, Callable
from src.helpers import PerceiverResampler
import open_clip

class ModalityFusioner(nn.Module):
    def __init__(self, 
                vision_encoder,
                num_history=10,
                 ):
        super(ModalityFusioner, self).__init__()
        self.language_x_dim = 768
        self.vs_x_dim = 768
        self.vg_x_dim = 768
        self.num_v_token = 49 * 2
        self.num_l_token = 8
        self.decoder_dim = 768
        #self.num_latents = 16
        self.num_history = num_history

        self.vision_encoder = vision_encoder

        self.vision_encoder.output_tokens = True
        self.vision_encoder.proj = None
        
        self.positional_encoding1 = PositionalEncoding(self.decoder_dim, dropout=0, max_len=self.num_v_token)

        #self.language_resampler = PerceiverResampler(dim=self.decoder_dim, depth=2, num_latents=self.num_latents)

        modality_fusion_layer = nn.TransformerDecoderLayer(d_model=self.decoder_dim, nhead=8, batch_first=True, dropout=0.1)
        self.modality_fusioner = nn.TransformerDecoder(modality_fusion_layer, num_layers=2)

        self.global_1d_pool = nn.AdaptiveMaxPool1d(1)
        
    def forward(self, rgb_static, rgb_gripper, language_x, language_padding_mask=None, obser_padding_mask=None):
        
        batch_size, num_history, vs_c, vs_h, vs_w = rgb_static.shape
        batch_size, num_history, vg_c, vg_h, vg_w = rgb_gripper.shape
        batch_size, num_l_token, ld = language_x.shape

        assert num_history == self.num_history, "input data length error"
        assert vs_h == 224 and vg_h == 224, "input data format error"
        assert num_l_token == self.num_l_token, "input language token error"

        rgb_static = rgb_static.view(batch_size*self.num_history, vs_c, vs_h, vs_w)
        rgb_gripper = rgb_gripper.view(batch_size*self.num_history, vg_c, vg_h, vg_w)

        vs_x = self.vision_encoder(rgb_static)[1]  # [(b t) vn vd]
        vg_x = self.vision_encoder(rgb_gripper)[1]  # [(b t) vn vd]

        v_x = torch.cat((vs_x, vg_x), dim=1) # [(b t) 2*vn vd]
        v_x = self.positional_encoding1(v_x)
        
        language_x = language_x.unsqueeze(1).repeat(1, self.num_history, 1, 1) # [b t ln ld]
        language_x = language_x.view(batch_size*self.num_history, self.num_l_token, self.decoder_dim) # [(b t) ln ld]
        
        x = self.modality_fusioner(tgt=language_x, memory=v_x) # [(b t) ln ld]

        x = rearrange(x, "bt ln ld -> bt ld ln") # [(b t) ld ln]
        x = self.global_1d_pool(x)# [(b t) ld 1]

        x = x.view(batch_size, self.num_history * self.decoder_dim) # [b t*ld]
        
        return x
        
class LanguageConnecter(nn.Module):
    def __init__(self):
        super(LanguageConnecter, self).__init__()
        self.num_latents = 8
        self.num_l_token = 16
        self.decoder_dim = 768
        self.language_resampler = PerceiverResampler(dim=self.decoder_dim, depth=2, num_latents=self.num_latents)
        
        
    def forward(self, language_x):
        
        batch_size, num_l_token, ld = language_x.shape

        assert num_l_token == self.num_l_token, "input language token error"

        
        language_x = language_x.unsqueeze(1).unsqueeze(1) # [b 1 1 ln ld]
        language_x = self.language_resampler(language_x) # [b 1 lan ld]
        language_x = language_x.view(batch_size, self.num_latents, ld)
        
        return language_x

from diffusion_modules import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

class DiffusionActionModel(nn.Module):
    def __init__(self,
            vision_encoder,
            pred_horizon = 16,
            obs_horizon = 2,
            obs_dim = 768,
            action_dim = 7,
            have_connector = True
        ):
        super(DiffusionActionModel, self).__init__()

        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.have_connector = have_connector

        self.nets = None
        if self.have_connector:
            self.nets = nn.ModuleDict({
                'modality_fusioner': ModalityFusioner(vision_encoder=vision_encoder, num_history=obs_horizon),
                'noise_pred_net': ConditionalUnet1D(
                    input_dim=action_dim,
                    global_cond_dim=obs_dim*obs_horizon
                ),
                'language_connecter': LanguageConnecter()
            })
        
        else:
            self.nets = nn.ModuleDict({
                'modality_fusioner': ModalityFusioner(vision_encoder=vision_encoder, num_history=obs_horizon),
                'noise_pred_net': ConditionalUnet1D(
                    input_dim=action_dim,
                    global_cond_dim=obs_dim*obs_horizon
                ),
            })
            
        num_diffusion_iters = 10
        self.num_diffusion_iters = num_diffusion_iters
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule='squaredcos_cap_v2',
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type='epsilon'
        )

    def forward(self, rgb_static, rgb_gripper, rel_actions, language_x):
        device = next(self.parameters()).device
        B = rgb_static.shape[0]

        # encoder vision features
        if self.have_connector:
            language_x = self.nets['language_connecter'](
                language_x
            )
        
        obs_cond = self.nets['modality_fusioner'](
            rgb_static, rgb_gripper, language_x
        )
        # (B, obs_horizon * obs_dim)

        # sample noise to add to actions
        noise = torch.randn(rel_actions.shape, device=device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()

        # add noise to the clean images according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_actions = self.noise_scheduler.add_noise(
            rel_actions, noise, timesteps) # [b t a]

        # predict the noise residual
        noise_pred = self.nets['noise_pred_net'](
            noisy_actions, timesteps, global_cond=obs_cond) # [b t a]
        
        return noise_pred, noise
    
    def infer(self, rgb_statics, rgb_grippers, language_x):
        obs_cond = self.nets['modality_fusioner'](
                        rgb_statics, rgb_grippers, language_x
                    )
        naction = torch.randn((1, self.pred_horizon, self.action_dim), device=0)

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

        action = naction[0]

        return action


if __name__ == "__main__":
    num_history = 10
    clip_vision_encoder_path: str = "ViT-B-32"
    clip_vision_encoder_pretrained: str = "openai"
    vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
        clip_vision_encoder_path, pretrained=clip_vision_encoder_pretrained
    )

    resume_training = False
    
    model = PolicyDecoder(                 
        num_history=num_history,
        vision_encoder=vision_encoder
    )
    #model.requires_grad_(False)
    model.eval()
    rgb_static = torch.rand(7,num_history,3,224,224)
    rgb_static[1,:4] = rgb_static[0,:4]
    
    rgb_gripper = torch.rand(7,num_history,3,224,224)
    rgb_gripper[1] = rgb_gripper[0]
    language_x = torch.rand(7,10,768)
    language_x[1:] = language_x[0]
    action, _ = model(rgb_static,rgb_gripper,language_x)

    print(action[1] - action[0])

    



