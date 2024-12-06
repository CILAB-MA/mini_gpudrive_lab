import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np


class GMM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, action_dim=3, n_components=10, time_dim=1):
        super(GMM, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.residual_block = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(4)
        ])
        
        self.head = nn.Linear(hidden_dim, n_components * (2 * action_dim + 1))
        self.n_components = n_components
        self.action_dim = action_dim
        self.time_dim = time_dim

    def get_gmm_params(self, x):
        """
        Get the parameters of the Gaussian Mixture Model
        """
        x = x.reshape(x.size(0), self.time_dim, x.size(-1))
        x = self.input_layer(x)
        
        for layer in self.residual_block:
            residual = x
            x = layer(x)
            x += residual
        
        params = self.head(x)
        
        means = params[..., :self.n_components * self.action_dim].view(-1, self.time_dim, self.n_components, self.action_dim)
        covariances = params[..., self.n_components * self.action_dim:2 * self.n_components * self.action_dim].view(-1, self.time_dim, self.n_components, self.action_dim)
        weights = params[..., -self.n_components:].view(-1, self.time_dim, self.n_components)
        
        covariances = torch.clamp(covariances, -20, 2)
        covariances = torch.exp(covariances)
        weights = torch.softmax(weights, dim=-1)
        
        return means, covariances, weights, self.n_components

    def forward(self, x, deterministic=None):
        """
        Sample actions from the Gaussian Mixture Model
        """
        means, covariances, weights, components = self.get_gmm_params(x)

        component_indices = torch.argmax(weights, dim=-1) if deterministic else dist.Categorical(weights).sample()
        component_indices = component_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, self.action_dim)
        
        sampled_means = torch.gather(means, 2, component_indices)
        sampled_covariances = torch.gather(covariances, 2, component_indices)
        
        actions = sampled_means if deterministic else dist.MultivariateNormal(sampled_means, torch.diag_embed(sampled_covariances)).sample()
        actions = actions.squeeze(2)
        
        # Squash actions and scaling
        actions = torch.tanh(actions)
        scale_factor = torch.tensor([6.0, 6.0, np.pi], device=actions.device)        
        actions = scale_factor * actions

        return actions
