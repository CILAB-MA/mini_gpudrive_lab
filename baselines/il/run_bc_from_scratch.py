"""Obtain a policy using behavioral cloning."""

# Torch
import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import os, sys, torch
sys.path.append(os.getcwd())
import wandb, yaml, argparse
from datetime import datetime
import numpy  as np
from tqdm import tqdm

# GPUDrive
from pygpudrive.env.config import EnvConfig, RenderConfig
from baselines.il.config import BehavCloningConfig
from algorithms.il.model.bc import *

def parse_args():
    parser = argparse.ArgumentParser('Select the dynamics model that you use')
    parser.add_argument('--dynamics-model', '-dm', type=str, default='delta_local', choices=['delta_local', 'bicycle', 'classic'],)
    parser.add_argument('--action-type', '-at', type=str, default='continuous', choices=['discrete', 'multi_discrete', 'continuous'],)
    parser.add_argument('--device', '-d', type=str, default='cuda', choices=['cpu', 'cuda'],)
    parser.add_argument('--model-name', '-m', type=str, default='bc_policy')
    parser.add_argument('--action-scale', '-as', type=int, default=100)
    parser.add_argument('--num-stack', '-s', type=int, default=5)
    parser.add_argument('--data-path', '-dp', type=str, default='/data')
    parser.add_argument('--data-file', '-df', type=str, default='new_train_trajectory_1000.npz')
    args = parser.parse_args()
    return args
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    args = parse_args()
    # Configurations
    env_config = EnvConfig(
        dynamics_model=args.dynamics_model,
        steer_actions = torch.round(torch.linspace(-0.3, 0.3, 7) * 1000) / 1000,
        accel_actions = torch.round(torch.linspace(-6.0, 6.0, 7) * 1000) / 1000,
        dx = torch.round(torch.linspace(-6.0, 6.0, 100) * 1000) / 1000,
        dy = torch.round(torch.linspace(-6.0, 6.0, 100) * 1000) / 1000,
        dyaw = torch.round(torch.linspace(-3.14, 3.14, 300) * 1000) / 1000,
    )
    render_config = RenderConfig()
    bc_config = BehavCloningConfig()

    # Get state action pairs
    expert_obs, expert_actions = [], []
    with np.load(os.path.join(args.data_path, args.data_file)) as npz:
        expert_obs.append(npz['obs'])
        expert_actions.append(npz['actions'])
    expert_obs = np.concatenate(expert_obs)
    expert_actions = np.concatenate(expert_actions)
    print(f'OBS SHAPE {expert_obs.shape} ACTIONS SHAPE {expert_actions.shape}')

    class ExpertDataset(torch.utils.data.Dataset):
        def __init__(self, obs, actions):
            self.obs = obs
            self.actions = actions

        def __len__(self):
            return len(self.obs)

        def __getitem__(self, idx):
            return self.obs[idx], self.actions[idx]

    # Make dataloader
    expert_dataset = ExpertDataset(expert_obs, expert_actions)
    expert_data_loader = DataLoader(
        expert_dataset,
        batch_size=bc_config.batch_size,
        shuffle=True,  # Break temporal structure
    )

    # # Build model
    bc_policy = ContFeedForwardMSE(
        input_size=expert_obs.shape[-1],
        hidden_size=bc_config.hidden_size,
        output_size=3,
    ).to(args.device)

    # Configure loss and optimizer
    optimizer = Adam(bc_policy.parameters(), lr=bc_config.lr)

    # Logging
    with open("private.yaml") as f:
        private_info = yaml.load(f, Loader=yaml.FullLoader)
    wandb.login(key=private_info["wandb_key"])
    currenttime = datetime.now().strftime("%Y%m%d%H%M%S")
    run_id = f"{type(bc_policy).__name__}_{currenttime}"
    wandb.init(
        project=private_info['main_project'],
        entity=private_info['entity'],
        name=run_id,
        id=run_id,
        group=f"{env_config.dynamics_model}_{args.action_type}",
        config={**bc_config.__dict__, **env_config.__dict__},
    )
    
    wandb.config.update({
        'lr': bc_config.lr,
        'batch_size': bc_config.batch_size,
        'num_stack': args.num_stack,
        'num_scene': expert_actions.shape[0],
        'num_vehicle': 128
    })
    
    global_step = 0
    for epoch in tqdm(range(bc_config.epochs), desc="Epochs"):
        for i, (obs, expert_action) in enumerate(expert_data_loader):

            obs, expert_action = obs.to(args.device), expert_action.to(
                args.device
            )

            # # Forward pass
            pred_action = bc_policy(obs)
            # mu, vars, mixed_weights = bc_policy(obs)
            # log_prob = bc_policy._log_prob(obs, expert_action)
            # loss = -log_prob
            loss = F.smooth_l1_loss(pred_action, expert_action * args.action_scale)
            # loss = gmm_loss(mu, vars, mixed_weights, expert_actions)
            # Backward pass
            with torch.no_grad():
                pred_action = bc_policy(obs)
                action_loss = torch.abs(pred_action - expert_action * args.action_scale) / args.action_scale
                dx_loss = action_loss[:, 0].mean().item()
                dy_loss = action_loss[:, 1].mean().item()
                dyaw_loss = action_loss[:, 2].mean().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log(
                {
                    "global_step": global_step,
                    "loss": loss.item(),
                    "dx_loss":dx_loss,
                    "dy_loss":dy_loss,
                    "dyaw_loss":dyaw_loss,
                }
            )

            global_step += 1

    # Save policy
    if bc_config.save_model:
        torch.save(bc_policy, f"{bc_config.model_path}/{args.model_name}.pth")
