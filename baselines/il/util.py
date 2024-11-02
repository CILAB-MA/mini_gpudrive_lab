import torch
import numpy as np
import os

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.01, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.verbose = verbose
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


# DATASET
class ExpertDataset(torch.utils.data.Dataset):
    def __init__(self, obs, actions):
        self.obs = obs
        self.actions = actions

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx]


def make_dataset(data_path):
    expert_obs, expert_actions = [], []
    
    data_files = os.listdir(data_path)
    data_files = [file for file in data_files if file.endswith('.npz')]
    data_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    for file in data_files:
        with np.load(os.path.join(data_path, file)) as npz:
            expert_obs.append(npz['obs'])
            expert_actions.append(npz['actions'])
    expert_obs = np.concatenate(expert_obs)
    expert_actions = np.concatenate(expert_actions)
    print(f'OBS SHAPE {expert_obs.shape} ACTIONS SHAPE {expert_actions.shape}')
    return expert_obs, expert_actions