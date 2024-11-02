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
from baselines.il.util import EarlyStopping, ExpertDataset, make_dataset
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
    parser.add_argument('--model-path', '-p', type=str, default='/data')
    args = parser.parse_args()
    return args
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == "__main__":
    args = parse_args()
    # Configurations
    env_config = EnvConfig(
        dynamics_model=args.dynamics_model,
    )
    render_config = RenderConfig()
    bc_config = BehavCloningConfig()

    # Load expert data
    train_obs, train_actions = make_dataset(args.data_path)
    dataset = ExpertDataset(train_obs, train_actions)
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

    # Set dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=bc_config.batch_size,
        shuffle=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=bc_config.batch_size,
        shuffle=True,
    )
    

    # # Build model
    bc_policy = ContFeedForwardMSE(
        input_size=train_obs.shape[-1],
        hidden_size=bc_config.hidden_size,
        output_size=3,
    ).to(args.device)

    # Configure loss and optimizer
    optimizer = Adam(bc_policy.parameters(), lr=bc_config.lr)

    # Logging
    with open("private.yaml") as f:
        private_info = yaml.load(f, Loader=yaml.FullLoader)
    wandb.login(key=private_info["my_wandb_key"])
    currenttime = datetime.now().strftime("%Y%m%d%H%M%S")
    cluster_path = os.path.basename(os.path.dirname(args.data_path))
    run_id = f"{type(bc_policy).__name__}_{currenttime}_{cluster_path}"
    wandb.init(
        project=private_info['my_project'],
        entity=private_info['my_entity'],
        name=run_id,
        id=run_id,
        group=f"{env_config.dynamics_model}_{args.action_type}",
        config={**bc_config.__dict__, **env_config.__dict__},
    )
    wandb.config.update({
        'lr': bc_config.lr,
        'batch_size': bc_config.batch_size,
        'num_stack': args.num_stack,
        'num_vehicle': 128
    })
    
    # Early stopping
    early_stopping = EarlyStopping(patience=5, min_delta=0.01, verbose=True)
    
    global_step = 0
    for epoch in tqdm(range(bc_config.epochs), desc="Epochs"):
        bc_policy.train()
        for i, (obs, expert_action) in enumerate(train_dataloader):
            obs, expert_action = obs.to(args.device), expert_action.to(
                args.device
            )

            # # Forward pass
            pred_action = bc_policy(obs)
            loss = F.mse_loss(pred_action, expert_action)
            # loss = F.smooth_l1_loss(pred_action, expert_action)
            action_loss = torch.abs(pred_action - expert_action)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log(
                {
                    "train/global_step": global_step,
                    "train/loss": loss.item(),
                    "train/dx_loss":action_loss[:, 0].mean().item(),
                    "train/dy_loss":action_loss[:, 1].mean().item(),
                    "train/dyaw_loss":action_loss[:, 2].mean().item(),
                }
            )
            global_step += 1

        bc_policy.eval()
        val_loss = 0.0
        val_dx_loss = 0.0
        val_dy_loss = 0.0
        val_dyaw_loss = 0.0
        with torch.no_grad():
            for obs, expert_action in valid_dataloader:
                obs, expert_action = obs.to(args.device), expert_action.to(args.device)
                pred_action = bc_policy(obs)
                val_loss += F.mse_loss(pred_action, expert_action).item()
                val_dx_loss += torch.abs(pred_action - expert_action)[:, 0].mean().item()
                val_dy_loss += torch.abs(pred_action - expert_action)[:, 1].mean().item()
                val_dyaw_loss += torch.abs(pred_action - expert_action)[:, 2].mean().item()
                
        val_loss /= len(valid_dataloader)
        val_dx_loss /= len(valid_dataloader)
        val_dy_loss /= len(valid_dataloader)
        val_dyaw_loss /= len(valid_dataloader)
        wandb.log(
            {
                "valid/epoch": epoch,
                "valid/loss": val_loss,
                "valid/dx_loss": val_dx_loss,
                "valid/dy_loss": val_dy_loss,
                "valid/dyaw_loss": val_dyaw_loss,
            }
        ) 
        
        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            break
        

    # Save policy
    if bc_config.save_model:
        torch.save(bc_policy, f"{args.model_path}/{args.model_name}.pth")
