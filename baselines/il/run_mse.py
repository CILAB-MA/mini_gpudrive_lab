"""Obtain a policy using behavioral cloning."""

# Torch
import logging
from torch.optim import Adam
from torch.utils.data import DataLoader
import os, sys
sys.path.append(os.getcwd())
import wandb, yaml, argparse
from datetime import datetime
import numpy as np
from tqdm import tqdm

# GPUDrive
from pygpudrive.env.config import EnvConfig
from baselines.il.config import BehavCloningConfig
from algorithms.il.model.bc import *


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser('Select the dynamics model that you use')
    parser.add_argument('--dynamics-model', '-dm', type=str, default='delta_local', choices=['delta_local', 'bicycle', 'classic'])
    parser.add_argument('--action-type', '-at', type=str, default='continuous', choices=['discrete', 'multi_discrete', 'continuous'])
    parser.add_argument('--device', '-d', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--model-name', '-m', type=str, default='bc_policy')
    parser.add_argument('--action-scale', '-as', type=int, default=100)
    parser.add_argument('--num-stack', '-s', type=int, default=5)
    parser.add_argument('--data-path', '-dp', type=str, default='/data')
    parser.add_argument('--scene-count', '-c', type=int, default=1)
    parser.add_argument('--use-wandb', '-w', action='store_true')
    args = parser.parse_args()
    return args


class ExpertDataset(torch.utils.data.Dataset):
    def __init__(self, obs, actions):
        self.obs = obs
        self.actions = actions

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx]


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = []
    for obs, expert_action in dataloader:
        obs, expert_action = obs.to(device), expert_action.to(device)
        pred_action = model(obs)
        action_loss = torch.abs(pred_action - expert_action).mean(dim=0)
        total_loss.append(action_loss)
    return torch.stack(total_loss).mean(dim=0)


if __name__ == "__main__":
    args = parse_args()

    # Configurations
    env_config = EnvConfig(
        dynamics_model=args.dynamics_model,
        steer_actions=torch.round(torch.linspace(-0.3, 0.3, 7) * 1000) / 1000,
        accel_actions=torch.round(torch.linspace(-6.0, 6.0, 7) * 1000) / 1000,
        dx=torch.round(torch.linspace(-6.0, 6.0, 100) * 1000) / 1000,
        dy=torch.round(torch.linspace(-6.0, 6.0, 100) * 1000) / 1000,
        dyaw=torch.round(torch.linspace(-3.14, 3.14, 300) * 1000) / 1000,
    )
    bc_config = BehavCloningConfig()

    # Get state action pairs
    expert_obs, expert_actions = [], []
    for file_name in tqdm(os.listdir(args.data_path), total=args.scene_count, desc="Loading data"):
        file_path = os.path.join(args.data_path, file_name)
        with np.load(file_path) as npz:
            expert_obs.append(npz['obs'])
            expert_actions.append(npz['actions'])
        if len(expert_obs) >= args.scene_count:
            break
    expert_obs = np.concatenate(expert_obs)
    expert_actions = np.concatenate(expert_actions)

    # Omit agent_id from observation
    num_stack = 5
    omit_idx = 7
    batch_size = expert_obs.shape[0]
    reshaped_obs = expert_obs.reshape(batch_size, num_stack, -1)
    omited_obs = np.concatenate([reshaped_obs[:, :, :omit_idx], reshaped_obs[:, :, omit_idx+1:]], axis=-1)
    expert_obs = omited_obs.reshape(batch_size, -1)

    print(f'OBS SHAPE {expert_obs.shape} ACTIONS SHAPE {expert_actions.shape}')

    # Make dataloader
    dataset = ExpertDataset(expert_obs, expert_actions)
    train_size = int(len(dataset) * 0.8)
    train_dataset, vali_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=bc_config.batch_size,
        shuffle=True,  # Break temporal structure
    )
    valid_dataloader = DataLoader(
        vali_dataset,
        batch_size=bc_config.batch_size,
        shuffle=False,
    )

    # Build model
    model = ContFeedForwardMSE(
        input_size=expert_obs.shape[-1],
        hidden_size=bc_config.hidden_size,
        output_size=expert_actions.shape[-1],
    ).to(args.device)

    # Configure loss and optimizer
    optimizer = Adam(model.parameters(), lr=bc_config.lr)

    # Logging
    if args.use_wandb:
        with open("private.yaml") as f:
            private_info = yaml.load(f, Loader=yaml.FullLoader)
        wandb.login(key=private_info["wandb_key"])
        currenttime = datetime.now().strftime("%Y%m%d%H%M%S")
        run_id = f"{type(model).__name__}_{currenttime}"
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

    train_losses, eval_losses = [], []
    for epoch in range(500):
        model.train()
        for i, (obs, expert_action) in enumerate(train_dataloader):
            obs, expert_action = obs.to(args.device), expert_action.to(args.device)

            # Forward pass
            pred_action = model(obs)

            # Calculate loss
            loss = F.smooth_l1_loss(pred_action, expert_action)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate metrics
            with torch.no_grad():
                action_loss = torch.abs(pred_action - expert_action).mean(dim=0)
            if args.use_wandb:
                wandb.log({
                    "global_step": epoch * len(train_dataloader) + i,
                    "loss": loss.item(),
                    "dx_loss": action_loss[0].item(),
                    "dy_loss": action_loss[1].item(),
                    "dyaw_loss": action_loss[2].item(),
                })

        eval_loss = evaluate(model, valid_dataloader, args.device)

        # Print results
        print(f"Epoch {epoch}: "
              f"train loss [{action_loss[0]:.5f}, {action_loss[1]:.5f}, {action_loss[2]:.5f}], "
              f"eval loss [{eval_loss[0]:.5f}, {eval_loss[1]:.5f}, {eval_loss[2]:.5f}]")
        train_losses.append(action_loss.cpu().numpy())
        eval_losses.append(eval_loss.cpu().numpy())

    # Plot the results
    import matplotlib.pyplot as plt
    train_losses = np.array(train_losses)
    eval_losses = np.array(eval_losses)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    for i, ax in enumerate(axes):
        ax.plot(train_losses[:, i], label='train')
        ax.plot(eval_losses[:, i], label='eval')
        ax.set_title(['dx', 'dy', 'dyaw'][i])
        ax.legend()
    plt.show()

    # Save policy
    if bc_config.save_model:
        os.makedirs(bc_config.model_path, exist_ok=True)
        torch.save(model, f"{bc_config.model_path}/{args.model_name}.pth")
