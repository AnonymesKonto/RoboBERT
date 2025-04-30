from diffusion_modules import ConditionalUnet1D
from torch import nn
import open_clip
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import torch

from data_enhence import my_mixup
from data_random import CalvinDataset
from pretrain_data_random import PretrainCalvinDataset
from ModalityFusioner import ModalityFusioner, LanguageConnecter, DiffusionActionModel
import numpy as np
from my_utils import get_parameter_number, remove_prefix

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import argparse

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '1355'
    
    dist.init_process_group(
        backend='nccl',  # 适用于 GPU 的后端
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)  # 设置当前进程对应的 GPU

def train(rank, args):
    print("it is DDP training")
    setup(rank, args.world_size)
    
    clip_vision_encoder_path: str = "ViT-B-32"
    clip_vision_encoder_pretrained: str = "openai"
    clip_encoder, _, image_processor = open_clip.create_model_and_transforms(
        clip_vision_encoder_path, pretrained=clip_vision_encoder_pretrained
    )
    vision_encoder = clip_encoder.visual
    
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 7
    obs_dim = 768
    action_dim = 7

    training_mode = args.training_mode
    num_epochs = -1
    dataset_name = args.dataset_name
    checkpoint = args.checkpoint

    ckpt_folder_path = "ckpt"
    if not os.path.exists(ckpt_folder_path):
        os.makedirs(ckpt_folder_path)

    if dataset_name not in {"ABC_D", "ABCD_D"}:
        raise("dataset_name error")
    
    if training_mode == "pretrain":
        print("training on the unlabelled data")
        num_epochs = 2

        model = DiffusionActionModel(
            vision_encoder = vision_encoder,
            pred_horizon = 16,
            obs_horizon = 2,
            obs_dim = 768,
            action_dim = 7,
            have_connector = False
        )

    elif training_mode == "first":
        print("training on the first stage")
        num_epochs = 10

        model = DiffusionActionModel(
            vision_encoder = vision_encoder,
            pred_horizon = 16,
            obs_horizon = 2,
            obs_dim = 768,
            action_dim = 7,
            have_connector = False
        )

        # checkpoint = torch.load("./ckpt/model-4-second-task_ABC_D-m2.pt")
        # model_state_dict = remove_prefix(checkpoint['model_state_dict'])
        # msg0 = model.load_state_dict(model_state_dict, strict=False)

        model.requires_grad_(True)
        model.nets["modality_fusioner"].vision_encoder.requires_grad_(False)
        model.nets["modality_fusioner"].vision_encoder.transformer.resblocks[11].requires_grad_(True)
        print(get_parameter_number(model))
         
    elif training_mode == "second":
        print("resuming the training mode, reading the checkpoint...")
        if dataset_name == "ABCD_D":
            num_epochs = 5
        if dataset_name == "ABC_D":
            num_epochs = 15
            
        model = DiffusionActionModel(
            vision_encoder = vision_encoder,
            pred_horizon = 16,
            obs_horizon = 2,
            obs_dim = 768,
            action_dim = 7,
            have_connector = True
        )

        checkpoint = torch.load(checkpoint)
        model_state_dict = remove_prefix(checkpoint['model_state_dict'])
        msg0 = model.load_state_dict(model_state_dict, strict=False)
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.requires_grad_(True)
        print(get_parameter_number(model))
        print(msg0)
    
    else:
        raise("Training Model Error")
        
    model = model.to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    # holds a copy of the model weights
    ema = EMAModel(
        parameters=model.parameters(),
        power=0.75)

    # Standard ADAM optimizer
    # Note that EMA parametesr are not optimized
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=2e-5, weight_decay=1e-6) #2e-5
        
    if False:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        optimizer.param_groups[0]['lr'] = 2e-6
    #print(checkpoint['optimizer_state_dict'])

    # set dataset and relevant

    if training_mode == "first" or training_mode == "second":
        dataset = CalvinDataset(
            dataset_name=dataset_name,
            image_processor=image_processor,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            training_mode=training_mode
        )

    if training_mode == "pretrain":
        dataset = PretrainCalvinDataset(
            dataset_path=f"/home/wibot/Data/SC/{dataset_name}",
            image_processor=image_processor,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
        )

    calvin_sampler = torch.utils.data.distributed.DistributedSampler(dataset) 

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        num_workers=4,
        shuffle=False,
        # accelerate cpu-gpu transfer
        pin_memory=False,
        # don't kill worker process afte each epoch
        sampler=calvin_sampler,
        persistent_workers=True
    )

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs
    )

    device = torch.device('cuda')
    _ = model.to(device)
    _ = ema.to(device)
    

    if rank == 0:
        log_tag = datetime.strftime(datetime.now(),'%Y-%m-%d-%H-%M-%S')
        writer = SummaryWriter(log_dir="./log/"+log_tag)

    global_step = 0
    seed_bias = 0
    if training_mode == "second":
        seed_bias = 100
        
    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            dataloader.sampler.set_epoch(epoch_idx + seed_bias)
            # batch loop
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    # data normalized in dataset
                    # device transfer
                    observe_space, task, language_x, padding_mask, action_space = nbatch

                    rgb_static = observe_space["rgb_static"] # [b obs_t 3 224 224]
                    rgb_gripper = observe_space["rgb_gripper"] # [b obs_t 3 224 224]
                    rel_actions = action_space["rel_actions"] # [b pred_t 7]
                    language_x = language_x # [b 16 768]
                    
                    rgb_static = rgb_static.to(device)
                    rgb_gripper = rgb_gripper.to(device)
                    rel_actions = rel_actions.to(device)
                    language_x = language_x.to(device)

                    if training_mode == "second" and dataset_name == "ABC_D":
                        rgb_static, rgb_gripper, rel_actions, language_x = my_mixup([rgb_static, rgb_gripper, rel_actions, language_x])
                    
                    noise_pred, noise = model(rgb_static, rgb_gripper, rel_actions, language_x)
 
                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise) #/ non_padding_ratio

                    # optimize
                    loss.backward()

                    # for name, param in model.named_parameters():
                    #     if param.grad is None:
                    #         print(name)
                    # print("###################################")

                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()

                    # update Exponential Moving Average of the model weights
                    # ema.step(nets.parameters())

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)

                    if rank == 0:
                        writer.add_scalar(tag="actor_loss_step", scalar_value=loss_cpu, global_step=global_step)
                    
                    global_step += 1
                #writer.add_scalar(tag="stat_loss_step", scalar_value=task_state_loss_v, global_step=global_step)
            tglobal.set_postfix(loss=np.mean(epoch_loss))
            if rank == 0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                    f"./ckpt/model-{epoch_idx}-{training_mode}-{dataset_name}.pt")
                print("saving complete!")
            
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='configurating the extraction specifications')
    parser.add_argument('--world_size', type=int, default=2, help="the number of GPUs in DDP training, assuming there are two GPUs")
    parser.add_argument('--training_mode', type=str, default="first", help="the training stage string, only support 'first' and 'second'")
    parser.add_argument('--dataset_name', type=str, default="ABCD_D")
    parser.add_argument('--checkpoint', type=str, default="path/to/ckpt", help="the checkpoint path for second stage training")
    args = parser.parse_args()
    torch.multiprocessing.spawn(train, args=(args,), nprocs=args.world_size, join=True)
