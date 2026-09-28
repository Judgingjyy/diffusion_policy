import torch


@torch.no_grad()
def sample_action(
    model,
    obs,
    alpha_bars,
    action_horizon,
    action_dim,
    num_diffusion_steps=100
):

    B = obs.shape[0]
    device = obs.device

    action = torch.randn(
        B,
        action_horizon,
        action_dim,
        device=device
    )

    for k in reversed(range(num_diffusion_steps)):

        k_batch = torch.full(
            (B,),
            k,
            device=device,
            dtype=torch.long
        )

        pred_noise = model(
            obs,
            action,
            k_batch
        )

        alpha_bar_k = alpha_bars[k]

        # ① 根据公式估计 clean action
        pred_clean_action =( action-torch.sqrt(1-alpha_bar_k)*pred_noise)*torch.sqrt(alpha_bar_k)
        if k in [99, 80, 50, 20, 0]:
            print(
                "k =", k,
                "action abs mean =",
                action.abs().mean().item()
            )

        # ② 如果 k == 0，已经结束
        if k == 0:
            action = pred_clean_action
            break

        alpha_bar_prev = alpha_bars[k - 1]

        # ③ 构造更干净的 A^(k-1)
        action = torch.sqrt(alpha_bar_prev)*pred_clean_action+torch.sqrt(1-alpha_bar_prev)*pred_noise

    return action

