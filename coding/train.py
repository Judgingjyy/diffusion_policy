import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import ChunkDataset
from model import DiffusionPolicyModel
from noise_scheduler import add_noise, alpha_bars

device=torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

dataset =ChunkDataset()

loader=DataLoader(
    dataset,
    batch_size=64,
    shuffle=True
)
model = DiffusionPolicyModel(
    obs_dim=2,
    action_horizon=4,
    action_dim=2,
    hidden_dim=64,
    num_diffusion_steps=100
).to(device)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

alpha_bars = alpha_bars.to(device)
for epoch in range(64):
    loss_total=0
    for obs,clean_action in loader:
        obs=obs.to(device)
        clean_action=clean_action.to(device)
        B=clean_action.shape[0]
        k = torch.randint(
        low=0,
        high=100,
        size=(B,),
        device=device
        )
        noise=torch.randn_like(clean_action)
        noisy_action=add_noise(
            clean_action,noise,k,alpha_bars
        )
        pred_noise=model(
            obs,
            noisy_action,
            k

        )
        loss=criterion(
            pred_noise,
            noise
        )
        loss_total+=loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"'Epoch: '{epoch} , Loss: {loss.item()/B}")

torch.save(
    model.state_dict(),
    "diffusion_policy.pt"
) 