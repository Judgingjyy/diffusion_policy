import torch
import torch.nn as nn


class DiffusionPolicyModel(nn.Module):

    def __init__(
        self,
        obs_dim,
        action_horizon,
        action_dim,
        hidden_dim=64,
        num_diffusion_steps=100
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.num_diffusion_steps = num_diffusion_steps

        action_flat_dim =action_horizon*action_dim

        input_dim =obs_dim+action_flat_dim+1

        output_dim = action_flat_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, output_dim)
        )
    def forward(
    self,
    obs,
    noisy_action,
    k
    ):
        noisy_action=noisy_action.flatten(start_dim=1)
        # 小简化
        k=k.float()
        k=k/(self.num_diffusion_steps-1)
        k=k.unsqueeze(1)
        x=torch.cat(
            [noisy_action,obs,k],dim=1)
        
        pred_noise=self.net(x)
        pred_noise = pred_noise.reshape(
            -1,
            self.action_horizon,
            self.action_dim
        )

        return pred_noise

model = DiffusionPolicyModel(
    obs_dim=2,
    action_horizon=4,
    action_dim=2
)
