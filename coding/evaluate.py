import torch

from model import DiffusionPolicyModel
from inference import sample_action
from noise_scheduler import alpha_bars
from dataset import ChunkDataset
from noise_scheduler import add_noise
from noise_scheduler import (
    betas,
    alphas,
    alpha_bars
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = DiffusionPolicyModel(
    obs_dim=2,
    action_horizon=4,
    action_dim=2,
    hidden_dim=64,
    num_diffusion_steps=100
).to(device)

model.load_state_dict(
    torch.load(
        "diffusion_policy.pt",
        map_location=device
    )
)

model.eval()

alpha_bars = alpha_bars.to(device)

dataset = ChunkDataset()

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    shuffle=False
)

obs, target_action = next(iter(loader))
obs = obs.to(device)
betas=betas.to(device)
alphas=alphas.to(device)
alpha_bars=alpha_bars
target_action = target_action.to(device)
pred_action = sample_action(
    model=model,
    obs=obs,
    betas=betas,
    alphas=alphas,
    alpha_bars=alpha_bars,
    action_horizon=4,
    action_dim=2,
    num_diffusion_steps=100,
    debug=True
)

mae = torch.mean(
    torch.abs(pred_action - target_action)
)

print("MAE:", mae.item())


