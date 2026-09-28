import torch

from model import DiffusionPolicyModel
from inference import sample_action
from noise_scheduler import alpha_bars
from dataset import ChunkDataset


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
target_action = target_action.to(device)

pred_action = sample_action(
    model,
    obs,
    alpha_bars,
    action_horizon=4,
    action_dim=2,
    num_diffusion_steps=100
)
mae = torch.mean(
    torch.abs(pred_action - target_action)
)

print("MAE:", mae.item())