from dataclasses import dataclass, field


@dataclass
class BehavCloningConfig:

    # Dataset & device
    data_dir: str = "/data"
    device: str = "cpu"

    # Number of scenarios / worlds
    num_worlds: int = 3
    max_cont_agents: int = 128

    # Discretize actions and use action indices
    discretize_actions: bool = True
    use_action_indices: bool = True
    # Record a set of trajectories as sanity check
    make_sanity_check_video: bool = True

    # Logging
    wandb_mode: str = "online"
    wandb_project: str = "il"
    log_interval: int = 500

    # Hyperparameters
    batch_size: int = 512
    epochs: int = 1000
    lr: float = 3e-4
    hidden_size: list = field(default_factory=lambda: [1024, 256])
    net_arch: list = field(default_factory=lambda: [64, 128])

    # Save policy
    save_model: bool = True
    model_path: str = "models/"
    model_name: str = "bc_policy"
