"""Obtain a policy using behavioral cloning."""
import logging
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.optim as optim
import os, sys, torch
sys.path.append(os.getcwd())
import wandb, yaml, argparse
from tqdm import tqdm
from datetime import datetime

# GPUDrive
from pygpudrive.env.config import EnvConfig
from baselines.il.config import ExperimentConfig
from algorithms.il import MODELS, LOSS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser('Select the dynamics model that you use')
    parser.add_argument('--action-type', '-at', type=str, default='continuous', choices=['discrete', 'multi_discrete', 'continuous'],)
    parser.add_argument('--device', '-d', type=str, default='cuda', choices=['cpu', 'cuda'],)
    parser.add_argument('--num-stack', '-s', type=int, default=1)
    
    # MODEL
    parser.add_argument('--model-path', '-mp', type=str, default='/data/model')
    parser.add_argument('--model-name', '-m', type=str, default='wayformer', choices=['bc', 'late_fusion', 'attention', 'wayformer'])
    parser.add_argument('--loss-name', '-l', type=str, default='gmm', choices=['l1', 'mse', 'twohot', 'nll', 'gmm'])
    parser.add_argument('--rollout-len', '-rl', type=int, default=10)
    parser.add_argument('--pred-len', '-pl', type=int, default=5)
    
    # DATA
    parser.add_argument('--data-path', '-dp', type=str, default='/data/wayformer/')
    parser.add_argument('--train-data-file', '-td', type=str, default='train_trajectory_1000.npz')
    parser.add_argument('--eval-data-file', '-ed', type=str, default='test_trajectory_200.npz')
    
    # EXPERIMENT
    parser.add_argument('--exp-name', '-en', type=str, default='all_data')
    args = parser.parse_args()
    
    return args

class ExpertDataset(torch.utils.data.Dataset):
    def __init__(self, obs, actions, masks=None, rollout_len=1, pred_len=1):
        self.obs = obs
        self.actions = actions
        self.masks = masks
        self.num_timestep = 1 if len(obs.shape) == 2 else obs.shape[1] - rollout_len - pred_len + 1
        self.rollout_len = rollout_len
        self.pred_len = pred_len
        self.num_timestep = 1 if len(obs.shape) == 2 else obs.shape[1] - rollout_len - pred_len + 1
        self.rollout_len = rollout_len
        self.pred_len = pred_len
        self.use_mask = False
        if self.masks is not None:
            self.use_mask = True
            if len(self.obs.shape) == 3:
                valid_indices = self.masks[..., None]
                self.obs = self.obs * valid_indices
                self.actions = self.actions * valid_indices #TODO: whether the mask operation is properly removing the intended elements.
            else:
                valid_indices = self.masks.flatten() == 0
                self.obs = self.obs.reshape(-1, self.obs.shape[-1])[valid_indices]
                self.actions = self.actions.reshape(-1, self.actions.shape[-1])[valid_indices]

    def __len__(self):
        return len(self.obs) * self.num_timestep

    def __getitem__(self, idx):
        # row, column -> 
        if self.num_timestep > 1:
            idx1 = idx // self.num_timestep
            idx2 = idx % self.num_timestep
            if self.use_mask:
                return self.obs[idx1, idx2:idx2 + self.rollout_len], \
            self.actions[idx1, idx2 + self.rollout_len:idx2 + self.rollout_len + self.pred_len], \
            self.masks[idx1 ,idx2:idx2 + self.rollout_len] #TODO: mask operation
            else:
                return self.obs[idx1, idx2:idx2 + self.rollout_len], \
            self.actions[idx1, idx2 + self.rollout_len:idx2 + self.rollout_len + self.pred_len]
        else:
            if self.use_mask:
                return self.obs[idx], self.actions[idx], self.masks[idx]
            else:
                return self.obs[idx], self.actions[idx]


if __name__ == "__main__":
    args = parse_args()
    exp_config = ExperimentConfig()
    env_config = EnvConfig(
        dynamics_model='delta_local',
        steer_actions=torch.round(
            torch.linspace(-0.3, 0.3, 7), decimals=3
        ),
        accel_actions=torch.round(
            torch.linspace(-6.0, 6.0, 7), decimals=3
        ),
        dx=torch.round(
            torch.linspace(-6.0, 6.0, 100), decimals=3
        ).to(args.device),
        dy=torch.round(
            torch.linspace(-6.0, 6.0, 100), decimals=3
        ).to(args.device),
        dyaw=torch.round(
            torch.linspace(-np.pi, np.pi, 100), decimals=3
        ).to(args.device),
    )
    
    # Get state action pairs
    train_expert_obs, train_expert_actions = [], []
    eval_expert_obs, eval_expert_actions, = [], []
    
    # Additional data depends on model
    train_expert_masks, eval_expert_masks = [], []
    
    with np.load(os.path.join(args.data_path, args.train_data_file)) as npz:
        train_expert_obs.append(npz['obs'])
        train_expert_actions.append(npz['actions'])
        if 'dead_mask' in npz.keys():
            train_expert_masks.append(npz['dead_mask'])
    with np.load(os.path.join(args.data_path, args.eval_data_file)) as npz:
        eval_expert_obs.append(npz['obs'])
        eval_expert_actions.append(npz['actions'])
        if 'dead_mask' in npz.keys():
            eval_expert_masks.append(npz['dead_mask'])

    train_expert_obs = np.concatenate(train_expert_obs)
    train_expert_actions = np.concatenate(train_expert_actions)
    train_expert_masks = np.concatenate(train_expert_masks) if len(train_expert_masks) > 0 else None

    eval_expert_obs = np.concatenate(eval_expert_obs)
    eval_expert_actions = np.concatenate(eval_expert_actions)
    eval_expert_masks = np.concatenate(eval_expert_masks) if (len(eval_expert_masks) > 0) else None

    # Make dataloader
    expert_dataset = ExpertDataset(train_expert_obs, train_expert_actions, train_expert_masks,
                                   rollout_len=args.rollout_len, pred_len=args.pred_len)
    expert_data_loader = DataLoader(
        expert_dataset,
        batch_size=exp_config.batch_size,
        shuffle=True,
    )
    eval_expert_dataset = ExpertDataset(eval_expert_obs, eval_expert_actions, eval_expert_masks,
                                   rollout_len=args.rollout_len, pred_len=args.pred_len)
    eval_expert_data_loader = DataLoader(
        eval_expert_dataset,
        batch_size=exp_config.batch_size,
        shuffle=False,
    )
    del train_expert_obs
    del train_expert_actions
    del train_expert_masks
    del eval_expert_obs
    del eval_expert_actions
    del eval_expert_masks
    
    # Build Model
    bc_policy = MODELS[args.model_name](env_config, exp_config, args.loss_name, args.num_stack).to(args.device)
    
    # Configure loss and optimizer
    optimizer = Adam(bc_policy.parameters(), lr=exp_config.lr)
    dataset_len = len(expert_dataset)

    # Logging
    with open("private.yaml") as f:
        private_info = yaml.load(f, Loader=yaml.FullLoader)
    wandb.login(key=private_info["wandb_key"])
    run_id = f"{type(bc_policy).__name__}_{args.exp_name}"
    model_save_path = f"{args.model_path}/{args.model_name}_{args.exp_name}.pth"
    wandb.init(
        project=private_info['main_project'],
        entity=private_info['entity'],
        name=run_id,
        id=run_id + "_" + datetime.now().strftime("%m%d%H%M"),
        group=f"{args.model_name}",
        config={**exp_config.__dict__, **env_config.__dict__},
        tags=[args.model_name, args.loss_name, args.exp_name, str(dataset_len)]
    )
    wandb.config.update({
        'num_stack': args.num_stack,
        'num_scene': dataset_len,
        'num_vehicle': 128,
        'model_save_path': model_save_path})
    
    global_step = 0
    masks = None
    for epoch in tqdm(range(exp_config.epochs), desc="Epochs", unit="epoch"):
        bc_policy.train()
        total_samples = 0
        losses = 0
        dx_losses = 0
        dy_losses = 0
        dyaw_losses = 0
        for i, batch in enumerate(expert_data_loader):
            # Check if adding this batch exceeds 50,000
            batch_size = batch[0].size(0)
            if total_samples + batch_size > exp_config.sample_per_epoch:
                break
            total_samples += batch_size
            
            # Data
            if len(batch) == 3:
                obs, expert_action, masks = batch
            else:
                obs, expert_action = batch
            obs, expert_action = obs.to(args.device), expert_action.to(args.device)
            masks = masks.to(args.device) if len(batch) == 3 else None
            
            # Forward pass
            loss = LOSS[args.loss_name](bc_policy, obs, expert_action, masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred_actions = bc_policy(obs, masks, deterministic=True)
                action_loss = torch.abs(pred_actions - expert_action)
                dx_loss = action_loss[..., 0].mean().item()
                dy_loss = action_loss[..., 1].mean().item()
                dyaw_loss = action_loss[..., 2].mean().item()
                dx_losses += dx_loss
                dy_losses += dy_loss
                dyaw_losses += dyaw_loss
                
            losses += loss.mean().item()
        
        # Log training losses
        wandb.log(
            {   
                "train/loss": losses / (i + 1),
                "train/loss": losses / (i + 1),
                "train/dx_loss": dx_losses / (i + 1),
                "train/dy_loss": dy_losses / (i + 1),
                "train/dyaw_loss": dyaw_losses / (i + 1),
            }
        )

        # Evaluation loop
        bc_policy.eval()
        total_samples = 0  # Initialize sample counter
        losses = 0
        dx_losses = 0
        dy_losses = 0
        dyaw_losses = 0
        for i, batch in enumerate(eval_expert_data_loader):
            # Check if adding this batch exceeds 50,000
            batch_size = batch[0].size(0)
            if total_samples + batch_size > int(exp_config.sample_per_epoch / 5): 
                break
            total_samples += batch_size
            
            # Data
            if len(batch) == 3:
                obs, expert_action, masks = batch
            else:
                obs, expert_action = batch
            obs, expert_action = obs.to(args.device), expert_action.to(args.device)
            masks = masks.to(args.device) if len(batch) == 3 else None
            
            with torch.no_grad():
                pred_actions = bc_policy(obs, masks, deterministic=True)
                action_loss = torch.abs(pred_actions - expert_action)
                dx_loss = action_loss[..., 0].mean().item()
                dy_loss = action_loss[..., 1].mean().item()
                dyaw_loss = action_loss[..., 2].mean().item()
                dx_losses += dx_loss
                dy_losses += dy_loss
                dyaw_losses += dyaw_loss
                losses += action_loss.mean().item()
            
        # Log evaluation losses
        wandb.log(
            {
                "eval/loss": losses / (i + 1) ,
                "eval/dx_loss": dx_losses / (i + 1),
                "eval/dy_loss": dy_losses / (i + 1),
                "eval/dyaw_loss": dyaw_losses / (i + 1),
            }
        )

    # Save policy
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    torch.save(bc_policy, f"{args.model_path}/{args.model_name}_{args.loss_name}_{args.exp_name}.pth")
