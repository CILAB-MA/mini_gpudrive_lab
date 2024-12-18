"""Obtain a policy using behavioral cloning."""
import logging
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, SubsetRandomSampler
import os, sys, torch
sys.path.append(os.getcwd())
import wandb, yaml, argparse, shutil
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
    parser.add_argument('--device', '-d', type=str, default='cpu', choices=['cpu', 'cuda'],)
    parser.add_argument('--num-stack', '-s', type=int, default=1)
    
    # MODEL
    parser.add_argument('--model-path', '-mp', type=str, default='/data/model')
    parser.add_argument('--model-name', '-m', type=str, default='wayformer', choices=['bc', 'late_fusion', 'attention', 'wayformer'])
    parser.add_argument('--loss-name', '-l', type=str, default='gmm', choices=['l1', 'mse', 'twohot', 'nll', 'gmm'])
    parser.add_argument('--rollout-len', '-rl', type=int, default=10)
    parser.add_argument('--pred-len', '-pl', type=int, default=5)
    
    # DATA
    parser.add_argument('--data-path', '-dp', type=str, default='.')
    parser.add_argument('--train-data-file', '-td', type=str, default='train_trajectory_1000.npz')
    parser.add_argument('--eval-data-file', '-ed', type=str, default='test_trajectory_200.npz')
    
    # EXPERIMENT
    parser.add_argument('--exp-name', '-en', type=str, default='all_data')
    parser.add_argument('--use-wandb', action='store_true')
    args = parser.parse_args()
    
    return args

class ExpertDataset(torch.utils.data.Dataset):
    def __init__(self, obs, actions, masks=None, other_info=None, road_mask=None,
                 rollout_len=1, pred_len=1):
        self.obs = obs
        obs_pad = np.zeros((obs.shape[0], rollout_len - 1, *obs.shape[2:]), dtype=np.float32)
        self.obs = np.concatenate([obs_pad, self.obs], axis=1)
        self.masks = 1 - masks
        dead_masks_pad = np.ones((self.masks.shape[0], rollout_len - 1, *self.masks.shape[2:]), dtype=np.float32)
        self.masks = np.concatenate([dead_masks_pad, self.masks], axis=1).astype('bool')

        self.road_mask = road_mask
        road_mask_pad = np.zeros((road_mask.shape[0], rollout_len - 1, *road_mask.shape[2:]), dtype=np.float32)
        self.road_mask = np.concatenate([road_mask_pad, self.road_mask], axis=1).astype('bool')
        
        self.actions = actions
        self.other_info = other_info
        self.num_timestep = 1 if len(obs.shape) == 2 else obs.shape[1] - rollout_len - pred_len + 2
        self.rollout_len = rollout_len
        self.pred_len = pred_len
        self.use_mask = False

        self.partner_mask = other_info[..., -1]
        partner_mask_pad = np.zeros((self.partner_mask.shape[0], rollout_len - 1, *self.partner_mask.shape[2:]), dtype=np.float32)
        self.partner_mask = np.concatenate([partner_mask_pad, self.partner_mask], axis=1).astype('bool')
        if self.masks is not None:
            self.use_mask = True
        self.valid_indices = self._compute_valid_indices()
        self.full_var = ['obs', 'actions', 'masks', 'partner_mask', 'road_mask'] # todo: add other info

    def __len__(self):
        return len(self.valid_indices)

    def _compute_valid_indices(self):
        N, T = self.masks.shape
        valid_time = np.arange(T - (self.rollout_len + self.pred_len - 2))
        valid_idx1, valid_idx2 = np.where(self.masks[:, valid_time + self.rollout_len + self.pred_len - 2] == 1)
        valid_idx2 = valid_time[valid_idx2]
        return list(zip(valid_idx1, valid_idx2))
    
    def __getitem__(self, idx):
        idx1, idx2 = self.valid_indices[idx]
        # row, column -> 
        batch = ()
        if self.num_timestep > 1:
            for var_name in self.full_var:
                if self.__dict__[var_name] is not None:
                    if var_name in ['obs', 'road_mask', 'partner_mask']:
                        data = self.__dict__[var_name][idx1, idx2:idx2 + self.rollout_len] # idx 0 -> (0, 0:10) -> (0, 9) end with first timestep
                    elif var_name == 'actions':
                        data = self.__dict__[var_name][idx1, idx2:idx2 + self.pred_len] # idx 0 -> (0, 0:5) -> start with first timestep
                    elif var_name == 'masks':
                        data = self.__dict__[var_name][idx1 ,idx2 + self.rollout_len + self.pred_len - 2] # idx 0 -> (0, 10 + 5 - 2) -> (0, 13) & padding = 9 -> end with last action timestep
                    else:
                        raise ValueError(f"Not in data {self.full_var}. Your input is {var_name}")
                    batch = batch + (data, )
                    if var_name == 'masks':
                        ego_mask_data = self.__dict__[var_name][idx1, idx2:idx2 + self.rollout_len]
                        batch = batch + (ego_mask_data, )
        else:
            for var_name in self.full_var:
                if self.__dict__[var_name] is not None:
                    data = self.__dict__[var_name][idx]
                    batch = batch + (data, )   
        return batch

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
    train_other_info, eval_other_info = [], []
    train_road_mask, eval_road_mask = [], []
    
    # Load cached data
    with np.load(os.path.join(args.data_path, args.train_data_file)) as npz:
        train_expert_obs = [npz['obs']]
        train_expert_actions = [npz['actions']]
        train_expert_masks = [npz['dead_mask']] if 'dead_mask' in npz.keys() else []
        train_other_info = [npz['other_info']] if 'other_info' in npz.keys() else []
        train_road_mask = [npz['road_mask']] if 'road_mask' in npz.keys() else []

    with np.load(os.path.join(args.data_path, args.eval_data_file)) as npz:
        eval_expert_obs = [npz['obs']]
        eval_expert_actions = [npz['actions']]
        eval_expert_masks = [npz['dead_mask']] if 'dead_mask' in npz.keys() else []
        eval_other_info = [npz['other_info']] if 'other_info' in npz.keys() else []
        eval_road_mask = [npz['road_mask']] if 'road_mask' in npz.keys() else []

    # Combine data (no changes)
    train_expert_obs = np.concatenate(train_expert_obs)
    train_expert_actions = np.concatenate(train_expert_actions)
    train_expert_masks = np.concatenate(train_expert_masks) if len(train_expert_masks) > 0 else None
    train_other_info = np.concatenate(train_other_info) if len(train_other_info) > 0 else None
    train_road_mask = np.concatenate(train_road_mask) if len(train_road_mask) > 0 else None

    eval_expert_obs = np.concatenate(eval_expert_obs)
    eval_expert_actions = np.concatenate(eval_expert_actions)
    eval_expert_masks = np.concatenate(eval_expert_masks) if len(eval_expert_masks) > 0 else None
    eval_other_info = np.concatenate(eval_other_info) if len(eval_other_info) > 0 else None
    eval_road_mask = np.concatenate(eval_road_mask) if len(eval_road_mask) > 0 else None
    num_cpus = os.cpu_count()
    train_dataset = ExpertDataset(
            train_expert_obs, train_expert_actions, train_expert_masks,
            other_info=train_other_info, road_mask=train_road_mask,
            rollout_len=args.rollout_len, pred_len=args.pred_len
        )
    dataset_len = len(train_dataset)
    # DataLoader with multiple workers
    expert_data_loader = DataLoader(
        train_dataset,
        batch_size=exp_config.batch_size,
        shuffle=True,
        num_workers=int(num_cpus / 2),
        pin_memory=True
    )

    eval_expert_data_loader = DataLoader(
        ExpertDataset(
            eval_expert_obs, eval_expert_actions, eval_expert_masks,
            other_info=eval_other_info, road_mask=eval_road_mask,
            rollout_len=args.rollout_len, pred_len=args.pred_len
        ),
        batch_size=exp_config.batch_size,
        shuffle=False,
        num_workers=int(num_cpus / 2),
        pin_memory=True
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
    if args.use_wandb:
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
            batch_size = batch[0].size(0)
            if total_samples + batch_size > exp_config.sample_per_epoch:
                break
            total_samples += batch_size
            
            # Data ['obs', 'actions', 'masks', 'ego_mask', 'partner_mask', 'road_mask']
            if len(batch) == 6:
                obs, expert_action, masks, ego_masks, partner_masks, road_masks = batch 
            elif len(batch) == 3:
                obs, expert_action, masks = batch
            else:
                obs, expert_action = batch
            obs, expert_action = obs.to(args.device), expert_action.to(args.device)
            masks = masks.to(args.device) if len(batch) > 2 else None
            ego_masks = ego_masks.to(args.device) if len(batch) > 3 else None
            partner_masks = partner_masks.to(args.device) if len(batch) > 3 else None
            road_masks = road_masks.to(args.device) if len(batch) > 3 else None
            all_masks= [masks, ego_masks, partner_masks, road_masks]
            # Forward pass
            loss = LOSS[args.loss_name](bc_policy, obs, expert_action, all_masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred_actions = bc_policy(obs, all_masks[1:], deterministic=True)
                action_loss = torch.abs(pred_actions - expert_action)
                dx_loss = action_loss[..., 0].mean().item()
                dy_loss = action_loss[..., 1].mean().item()
                dyaw_loss = action_loss[..., 2].mean().item()
                dx_losses += dx_loss
                dy_losses += dy_loss
                dyaw_losses += dyaw_loss
                
            losses += loss.mean().item()
        if args.use_wandb:
            # Log training losses
            wandb.log(
                {   
                    "train/loss": losses / (i + 1),
                    "train/loss": losses / (i + 1),
                    "train/dx_loss": dx_losses / (i + 1),
                    "train/dy_loss": dy_losses / (i + 1),
                    "train/dyaw_loss": dyaw_losses / (i + 1),
                }, step=epoch
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
            
            # Data #todo: add other info
            if len(batch) == 6:
                obs, expert_action, masks, ego_masks, partner_masks, road_masks = batch  
            elif len(batch) == 3:
                obs, expert_action, masks = batch
            else:
                obs, expert_action = batch
            obs, expert_action = obs.to(args.device), expert_action.to(args.device)
            masks = masks.to(args.device) if len(batch) > 2 else None
            ego_masks = ego_masks.to(args.device) if len(batch) > 3 else None
            partner_masks = partner_masks.to(args.device) if len(batch) > 3 else None
            road_masks = road_masks.to(args.device) if len(batch) > 3 else None
            all_masks= [masks, ego_masks, partner_masks, road_masks]
            
            with torch.no_grad():
                pred_actions = bc_policy(obs, all_masks[1:], deterministic=True)
                action_loss = torch.abs(pred_actions - expert_action)
                dx_loss = action_loss[..., 0].mean().item()
                dy_loss = action_loss[..., 1].mean().item()
                dyaw_loss = action_loss[..., 2].mean().item()
                dx_losses += dx_loss
                dy_losses += dy_loss
                dyaw_losses += dyaw_loss
                losses += action_loss.mean().item()
        if args.use_wandb:  
        # Log evaluation losses
            wandb.log(
                {
                    "eval/loss": losses / (i + 1) ,
                    "eval/dx_loss": dx_losses / (i + 1),
                    "eval/dy_loss": dy_losses / (i + 1),
                    "eval/dyaw_loss": dyaw_losses / (i + 1),
                }, step=epoch
            )

    # Save policy
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    torch.save(bc_policy, f"{args.model_path}/{args.model_name}_{args.loss_name}_{args.exp_name}.pth")
