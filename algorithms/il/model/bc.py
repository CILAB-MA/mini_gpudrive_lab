# Define network
import torch.nn as nn
import torch
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import torch.nn.functional as F

import math

# Define network
class FeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForward, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_size, output_size)])

    def dist(self, obs):
        """Generate action distribution."""
        x_out = self.nn(obs.float())
        return [Categorical(logits=head(x_out)) for head in self.heads]

    def forward(self, obs, deterministic=False):
        """Generate an output from tensor input."""
        action_dist = self.dist(obs)

        if deterministic:
            actions_idx = action_dist[0].logits.argmax(axis=-1)
        else:
            actions_idx = action_dist.sample()
        return actions_idx

    def _log_prob(self, obs, expert_actions):
        pred_action_dist = self.dist(obs)
        log_prob = pred_action_dist[0].log_prob(expert_actions).mean()
        return log_prob

# Define network
class ContFeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ContFeedForward, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(input_size, hidden_size[0]),
            nn.ReLU(),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(),
            nn.Linear(hidden_size[1], hidden_size[1]),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size[1], hidden_size[1]),
                nn.ReLU(),
                nn.Linear(hidden_size[1], 1),
            )
            for _ in range(output_size)
        ])
        self.log_std = nn.Parameter(torch.zeros(output_size))

    def dist(self, obs):
        """Generate action distribution."""
        x_out = self.nn(obs.float())
        return [Normal(head(x_out), torch.exp(std)) for head, std in zip(self.heads, self.log_std)]

    def forward(self, obs, deterministic=False):
        """Generate an output from tensor input."""
        action_dist = self.dist(obs)
        if deterministic:
            actions_idx = torch.cat([dist.mean for dist in action_dist], dim=-1)
        else:
            actions_idx = torch.cat([dist.sample() for dist in action_dist], dim=-1)
        return actions_idx

    def _log_prob(self, obs, expert_actions):
        pred_action_dist = self.dist(obs)
        log_prob = torch.cat([
            dist.log_prob(action.unsqueeze(-1))
            for action, dist in zip(expert_actions.T, pred_action_dist)
        ], dim=-1).sum()
        return log_prob

# Define network
class ContSharedFeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_stack):
        super(ContSharedFeedForward, self).__init__()
        self.num_stack = num_stack
        self.encoder = nn.Sequential(
            nn.Linear(input_size // num_stack, hidden_size[0] // num_stack),
            nn.ReLU(),
        )
        self.nn = nn.Sequential(
            nn.Linear(hidden_size[0] // num_stack * num_stack, hidden_size[1]),
            nn.ReLU(),
            nn.Linear(hidden_size[1], hidden_size[1]),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size[1], hidden_size[1]),
                nn.ReLU(),
                nn.Linear(hidden_size[1], 1),
            )
            for _ in range(output_size)
        ])
        self.log_std = nn.Parameter(torch.zeros(output_size))

    def dist(self, obs):
        """Generate action distribution."""
        reshaped_obs = obs.view(obs.size(0), self.num_stack, -1)
        encoded_obs = self.encoder(reshaped_obs)
        concatenated_obs = encoded_obs.view(obs.size(0), -1)
        x_out = self.nn(concatenated_obs)
        return [Normal(head(x_out), torch.exp(std)) for head, std in zip(self.heads, self.log_std)]

    def forward(self, obs, deterministic=False):
        """Generate an output from tensor input."""
        action_dist = self.dist(obs)
        if deterministic:
            actions_idx = torch.cat([dist.mean for dist in action_dist], dim=-1)
        else:
            actions_idx = torch.cat([dist.sample() for dist in action_dist], dim=-1)
        return actions_idx

    def _log_prob(self, obs, expert_actions):
        pred_action_dist = self.dist(obs)
        log_prob = torch.cat([
            dist.log_prob(action.unsqueeze(-1))
            for action, dist in zip(expert_actions.T, pred_action_dist)
        ], dim=-1).sum()
        return log_prob

# Define network
class ContFeedForwardMSE(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ContFeedForwardMSE, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(input_size, hidden_size[0]),
            nn.ReLU(),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(),
            nn.Linear(hidden_size[1], hidden_size[1]),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size[1], hidden_size[1]),
                nn.ReLU(),
                nn.Linear(hidden_size[1], 1),
            )
            for _ in range(output_size)
        ])

    def forward(self, obs, deterministic=False):
        """Generate an output from tensor input."""
        nn = self.nn(obs)
        actions = torch.cat([head(nn) for head in self.heads], dim=-1)
        return actions

# Define network
class ContSharedFeedForwardMSE(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_stack):
        super(ContSharedFeedForwardMSE, self).__init__()
        self.num_stack = num_stack
        self.encoder = nn.Sequential(
            nn.Linear(input_size // num_stack, hidden_size[0] // num_stack),
            nn.ReLU(),
        )
        self.nn = nn.Sequential(
            nn.Linear(hidden_size[0] // num_stack * num_stack, hidden_size[1]),
            nn.ReLU(),
            nn.Linear(hidden_size[1], hidden_size[1]),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size[1], hidden_size[1]),
                nn.ReLU(),
                nn.Linear(hidden_size[1], 1),
            )
            for _ in range(output_size)
        ])

    def forward(self, obs, deterministic=False):
        """Generate an output from tensor input."""
        reshaped_obs = obs.view(obs.size(0), self.num_stack, -1)
        encoded_obs = self.encoder(reshaped_obs)
        concatenated_obs = encoded_obs.view(obs.size(0), -1)
        nn = self.nn(concatenated_obs)
        actions = torch.cat([head(nn) for head in self.heads], dim=-1)
        return actions

# Define network
class ContFeedForwardConv1D(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ContFeedForwardConv1D, self).__init__()
        
        # Define Conv1d layers
        self.conv_nn = nn.Sequential(
            nn.Conv1d(1, hidden_size[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size[0], hidden_size[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size[1], hidden_size[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Compute the output size of the conv layers
        conv_output_size = self.calculate_conv_output_size(input_size, 3, 1, 1)

        # Define output heads using conv_output_size
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size[1] * conv_output_size, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
            for _ in range(output_size)
        ])

    def calculate_conv_output_size(self, input_size, kernel_size, stride, padding):
        return math.floor((input_size + 2 * padding - kernel_size) / stride + 1)

    def forward(self, obs, deterministic=False):
        obs = obs.unsqueeze(1)  # Add channel dimension
        conv_out = self.conv_nn(obs)  # Shape: (batch_size, hidden_size[1], sequence_length)
        conv_out_flat = conv_out.view(conv_out.size(0), -1)  # Flatten for Linear layers
        actions = torch.cat([head(conv_out_flat) for head in self.heads], dim=-1)
        return actions
    
# Define network
class ContFeedForwardConv2D(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        input_size: Tuple (sequence_length, stack_num, feature)
        hidden_size: List of hidden channel sizes for Conv2D layers
        output_size: Number of output heads
        """
        super(ContFeedForwardConv2D, self).__init__()
        
        sequence_length, stack_num, feature = input_size

        # Define the 2D convolutional backbone
        self.conv_nn = nn.Sequential(
            nn.Conv2d(stack_num, hidden_size[0], kernel_size=(3, 3), stride=1, padding=1),  # Conv layer 1
            nn.ReLU(),
            nn.Conv2d(hidden_size[0], hidden_size[1], kernel_size=(3, 3), stride=1, padding=1),  # Conv layer 2
            nn.ReLU(),
            nn.Conv2d(hidden_size[1], hidden_size[1], kernel_size=(3, 3), stride=1, padding=1),  # Conv layer 3
            nn.ReLU(),
        )

        # Calculate the output size after Conv2D
        self.flattened_size = hidden_size[1] * sequence_length * feature

        # Define output heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.flattened_size, hidden_size[1]),
                nn.ReLU(),
                nn.Linear(hidden_size[1], 1),
            )
            for _ in range(output_size)
        ])

    def forward(self, obs, deterministic=False):
        """
        Generate an output from tensor input.
        Assumes obs is of shape [batch_size, stack_num, sequence_length, feature].
        """
        # Pass through Conv2D layers
        conv_out = self.conv_nn(obs)  # Shape: (batch_size, hidden_channels, sequence_length, feature)
        
        # Flatten for fully connected layers
        conv_out_flat = conv_out.view(conv_out.size(0), -1)  # Shape: (batch_size, flattened_size)
        
        # Generate outputs for all heads
        actions = torch.cat([head(conv_out_flat) for head in self.heads], dim=-1)
        return actions