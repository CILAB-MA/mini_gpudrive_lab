# Define network
from typing import List
import torch.nn as nn
import torch
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import torch.nn.functional as F

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
class ContLateFusionMSE(nn.Module):
    EGO_FEAT_DIM = 6
    PARTNER_FEAT_DIM = 10
    ROAD_GRAPH_FEAT_DIM = 13

    def __init__(
        self, env_config: dict, net_config: dict
    ):
        super(ContLateFusionMSE, self).__init__()

        # Unpack feature dimensions
        self.ego_input_dim = self.EGO_FEAT_DIM if env_config["ego_state"] else 0
        self.ro_input_dim = self.PARTNER_FEAT_DIM if env_config["partner_obs"] else 0
        self.rg_input_dim = self.ROAD_GRAPH_FEAT_DIM if env_config["road_map_obs"] else 0
        self.ro_max = env_config["max_num_agents_in_scene"] - 1
        self.rg_max = env_config["roadgraph_top_k"]
        self.num_stack = env_config["num_stack"]

        # Network architectures
        self.act_func = (
            nn.Tanh() if net_config["act_func"] == "tanh" else nn.ReLU()
        )
        self.dropout = net_config["dropout"]

        # Scene encoder
        self.ego_state_net = self._build_network(
            input_dim=self.ego_input_dim * self.num_stack,
            net_arch=net_config["ego_state_layers"],
        )
        self.road_object_net = self._build_network(
            input_dim=self.ro_input_dim * self.num_stack,
            net_arch=net_config["road_object_layers"],
        )
        self.road_graph_net = self._build_network(
            input_dim=self.rg_input_dim * self.num_stack,
            net_arch=net_config["road_graph_layers"],
        )
        self.head = self._build_out_network(
            input_dim=(
                net_config["ego_state_layers"][-1]
                + net_config["road_object_layers"][-1]
                + net_config["road_graph_layers"][-1]
            ),
            output_dim=net_config["last_layer_dim_pi"],
            net_arch=net_config["shared_layers"],
        )

    def _unpack_obs(self, obs_flat):
        """
        Unpack the flattened observation into the ego state and visible state.
        Args:
            obs_flat (torch.Tensor): flattened observation tensor of shape (batch_size, obs_dim)
        Return:
            ego_state, road_objects, stop_signs, road_graph (torch.Tensor).
        """
        ego_size = self.ego_input_dim
        ro_size = self.ro_input_dim * self.ro_max
        rg_size = self.rg_input_dim * self.rg_max
        obs_flat_unstack = obs_flat.reshape(-1, self.num_stack, ego_size + ro_size + rg_size)
        ego_stack = (
            obs_flat_unstack[..., :ego_size]
            .view(-1, self.num_stack, self.ego_input_dim)
            .reshape(-1, self.num_stack * self.ego_input_dim)
        )
        ro_stack = (
            obs_flat_unstack[..., ego_size:ego_size + ro_size]
            .view(-1, self.num_stack, self.ro_max, self.ro_input_dim)
            .permute(0, 2, 1, 3)  # Reorder to (batch, ro_max, num_stack, ro_input_dim)
            .reshape(-1, self.ro_max, self.num_stack * self.ro_input_dim)
        )

        # rg_stack: Original reshape, then combine num_stack and self.rg_input_dim dimensions
        rg_stack = (
            obs_flat_unstack[..., ego_size + ro_size: ego_size + ro_size + rg_size]
            .view(-1, self.num_stack, self.rg_max, self.rg_input_dim)
            .permute(0, 2, 1, 3)  # Reorder to (batch, rg_max, num_stack, rg_input_dim)
            .reshape(-1, self.rg_max, self.num_stack * self.rg_input_dim)
        )

        return ego_stack, ro_stack, rg_stack

    def _build_network(
        self, input_dim: int, net_arch: List[int],
    ) -> nn.Module:
        """Build a network with the specified architecture."""
        layers = []
        last_dim = input_dim
        for layer_dim in net_arch:
            layers.append(nn.Linear(last_dim, layer_dim))
            layers.append(nn.Dropout(self.dropout))
            layers.append(nn.LayerNorm(layer_dim))
            layers.append(self.act_func)
            last_dim = layer_dim
        return nn.Sequential(*layers)

    def _build_out_network(
        self, input_dim: int, output_dim: int, net_arch: List[int]
    ) -> nn.Module:
        """Create the output network architecture."""
        layers = []
        prev_dim = input_dim
        for layer_dim in net_arch:
            layers.append(nn.Linear(prev_dim, layer_dim))
            layers.append(nn.LayerNorm(layer_dim))
            layers.append(self.act_func)
            layers.append(nn.Dropout(self.dropout))
            prev_dim = layer_dim

        # Add final layer
        layers.append(nn.Linear(prev_dim, output_dim))

        return nn.Sequential(*layers)

    def get_embedded_obs(self, obs):
        """Get the embedded observation."""
        ego_state, road_objects, road_graph = self._unpack_obs(obs)
        ego_state = self.ego_state_net(ego_state)
        road_objects = self.road_object_net(road_objects)
        road_graph = self.road_graph_net(road_graph)

        # Max pooling across the object dimension
        # (M, E) -> (1, E) (max pool across features)
        road_objects = F.max_pool1d(
            road_objects.permute(0, 2, 1), kernel_size=self.ro_max
        ).squeeze(-1)
        road_graph = F.max_pool1d(
            road_graph.permute(0, 2, 1), kernel_size=self.rg_max
        ).squeeze(-1)

        embedding_vector = torch.cat((ego_state, road_objects, road_graph), dim=1)
        return embedding_vector

    def forward(self, obss):
        # Unpack observation
        embedding_vector = self.get_embedded_obs(obss)
        actions = self.head(embedding_vector)

        return actions

# Define network
class ContSharedLateFusionMSE(nn.Module):
    EGO_FEAT_DIM = 6
    PARTNER_FEAT_DIM = 10
    ROAD_GRAPH_FEAT_DIM = 13

    def __init__(
        self, env_config: dict, net_config: dict
    ):
        super(ContSharedLateFusionMSE, self).__init__()

        # Unpack feature dimensions
        self.ego_input_dim = self.EGO_FEAT_DIM if env_config["ego_state"] else 0
        self.ro_input_dim = self.PARTNER_FEAT_DIM if env_config["partner_obs"] else 0
        self.rg_input_dim = self.ROAD_GRAPH_FEAT_DIM if env_config["road_map_obs"] else 0
        self.ro_max = env_config["max_num_agents_in_scene"] - 1
        self.rg_max = env_config["roadgraph_top_k"]
        self.num_stack = env_config["num_stack"]

        # Network architectures
        self.act_func = (
            nn.Tanh() if net_config["act_func"] == "tanh" else nn.ReLU()
        )
        self.dropout = net_config["dropout"]

        # Build the networks
        # Actor network
        self.ego_state_net = self._build_network(
            input_dim=self.ego_input_dim,
            net_arch=net_config["ego_state_layers"],
        )
        self.road_object_net = self._build_network(
            input_dim=self.ro_input_dim,
            net_arch=net_config["road_object_layers"],
        )
        self.road_graph_net = self._build_network(
            input_dim=self.rg_input_dim,
            net_arch=net_config["road_graph_layers"],
        )
        self.head = self._build_out_network(
            input_dim=(
                net_config["ego_state_layers"][-1]
                + net_config["road_object_layers"][-1]
                + net_config["road_graph_layers"][-1]
            ) * self.num_stack,
            output_dim=net_config["last_layer_dim_pi"],
            net_arch=net_config["shared_layers"],
        )

    def _unpack_obs(self, obs_flat):
        """
        Unpack the flattened observation into the ego state and visible state.
        Args:
            obs_flat (torch.Tensor): flattened observation tensor of shape (batch_size, obs_dim)
        Return:
            ego_state, road_objects, stop_signs, road_graph (torch.Tensor).
        """
        ego_size = self.ego_input_dim
        ro_size = self.ro_input_dim * self.ro_max
        rg_size = self.rg_input_dim * self.rg_max
        obs_flat_unstack = obs_flat.reshape(-1, self.num_stack, ego_size + ro_size + rg_size)
        ego_stack = (
            obs_flat_unstack[..., :ego_size]
            .view(-1, self.num_stack, self.ego_input_dim)
            # .reshape(-1, self.num_stack * self.ego_input_dim)
        )
        ro_stack = (
            obs_flat_unstack[..., ego_size:ego_size + ro_size]
            .view(-1, self.num_stack, self.ro_max, self.ro_input_dim)
            .permute(0, 2, 1, 3)  # Reorder to (batch, ro_max, num_stack, ro_input_dim)
            # .reshape(-1, self.ro_max, self.num_stack * self.ro_input_dim)
        )

        # rg_stack: Original reshape, then combine num_stack and self.rg_input_dim dimensions
        rg_stack = (
            obs_flat_unstack[..., ego_size + ro_size: ego_size + ro_size + rg_size]
            .view(-1, self.num_stack, self.rg_max, self.rg_input_dim)
            .permute(0, 2, 1, 3)  # Reorder to (batch, rg_max, num_stack, rg_input_dim)
            # .reshape(-1, self.rg_max, self.num_stack * self.rg_input_dim)
        )

        return ego_stack, ro_stack, rg_stack

    def _build_network(
        self, input_dim: int, net_arch: List[int],
    ) -> nn.Module:
        """Build a network with the specified architecture."""
        layers = []
        last_dim = input_dim
        for layer_dim in net_arch:
            layers.append(nn.Linear(last_dim, layer_dim))
            layers.append(nn.Dropout(self.dropout))
            layers.append(nn.LayerNorm(layer_dim))
            layers.append(self.act_func)
            last_dim = layer_dim
        return nn.Sequential(*layers)

    def _build_out_network(
        self, input_dim: int, output_dim: int, net_arch: List[int]
    ) -> nn.Module:
        """Create the output network architecture."""
        layers = []
        prev_dim = input_dim
        for layer_dim in net_arch:
            layers.append(nn.Linear(prev_dim, layer_dim))
            layers.append(nn.LayerNorm(layer_dim))
            layers.append(self.act_func)
            layers.append(nn.Dropout(self.dropout))
            prev_dim = layer_dim

        # Add final layer
        layers.append(nn.Linear(prev_dim, output_dim))

        return nn.Sequential(*layers)

    def get_embedded_obs(self, obs):
        """Get the embedded observation."""
        ego_state, road_objects, road_graph = self._unpack_obs(obs)
        ego_state = self.ego_state_net(ego_state)
        road_objects = self.road_object_net(road_objects)
        road_graph = self.road_graph_net(road_graph)

        # Flatten the tensors
        ego_state = ego_state.view(*ego_state.shape[:1], -1)
        road_objects = road_objects.view(*road_objects.shape[:2], -1)
        road_graph = road_graph.view(*road_graph.shape[:2], -1)

        # Max pooling across the object dimension
        # (M, E) -> (1, E) (max pool across features)
        road_objects = F.max_pool1d(
            road_objects.permute(0, 2, 1), kernel_size=self.ro_max
        ).squeeze(-1)
        road_graph = F.max_pool1d(
            road_graph.permute(0, 2, 1), kernel_size=self.rg_max
        ).squeeze(-1)

        embedding_vector = torch.cat((ego_state, road_objects, road_graph), dim=1)
        return embedding_vector

    def forward(self, obss):
        # Unpack observation
        embedding_vector = self.get_embedded_obs(obss)
        actions = self.head(embedding_vector)

        return actions
