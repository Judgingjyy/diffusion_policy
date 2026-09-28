import torch

num_diffusion_steps=100

betas=torch.linspace(
    0.0001,
    0.02,
    num_diffusion_steps
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

