import torch
import math
num_diffusion_steps=100
s=0.008
# betas=torch.linspace(
#     0.0001,
#     0.02,
#     num_diffusion_steps
# )

steps=torch.linspace(
    0,
    num_diffusion_steps,
    num_diffusion_steps+1
)
f = torch.cos(
    (
        steps / num_diffusion_steps + s
    )
    /
    (1 + s)
    *
    math.pi / 2
) ** 2
alpha_bars_full = f / f[0]
betas = 1 - (
    alpha_bars_full[1:]
    /
    alpha_bars_full[:-1]
)
betas = torch.clamp(
    betas,
    max=0.999
)
alphas=1-betas
alpha_bars=torch.cumprod(
    alphas,
    dim=0
)

# print(alpha_bars[0])
# print(alpha_bars[99])

def add_noise(
        clean_action,
        noise,
        k,# k是tensor
        alpha_bars
):
    alpha_bar_k=alpha_bars[k]
    alpha_bar_k=alpha_bar_k.reshape(-1,1,1)
    Ar=torch.sqrt(alpha_bar_k)*clean_action
    Nn=torch.sqrt(1-alpha_bar_k)*noise
    return Ar+Nn

