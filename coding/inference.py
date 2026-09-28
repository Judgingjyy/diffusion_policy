import torch


@torch.no_grad()
def sample_action(
    model,
    obs,
    betas,
    alphas,
    alpha_bars,
    action_horizon=4,
    action_dim=2,
    num_diffusion_steps=100,
    debug=False
):
    """
    obs:
        [B, obs_dim]

    return:
        action: [B, action_horizon, action_dim]
    """

    device = obs.device
    B = obs.shape[0]

    # 保证 noise schedule 和模型在同一个 device
    betas = betas.to(device)
    alphas = alphas.to(device)
    alpha_bars = alpha_bars.to(device)

    # -----------------------------------
    # 1. 从纯 Gaussian noise 开始
    # A^K ~ N(0, I)
    # -----------------------------------
    action = torch.randn(
        B,
        action_horizon,
        action_dim,
        device=device
    )

    # -----------------------------------
    # 2. K-1 -> ... -> 0
    # -----------------------------------
    for k in reversed(range(num_diffusion_steps)):

        # 模型要求 k 的 shape 是 [B]
        k_batch = torch.full(
            (B,),
            k,
            device=device,
            dtype=torch.long
        )

        # ε_theta(O, A^k, k)
        pred_noise = model(
            obs,
            action,
            k_batch
        )

        beta_k = betas[k]
        alpha_k = alphas[k]
        alpha_bar_k = alpha_bars[k]

        # -----------------------------------
        # DDPM reverse mean
        #
        # μ =
        # 1 / sqrt(alpha_k)
        # *
        # (
        #     A^k
        #     -
        #     beta_k / sqrt(1-alpha_bar_k)
        #     * predicted_noise
        # )
        # -----------------------------------
        mean = (
            1.0 / torch.sqrt(alpha_k)
        ) * (
            action
            -
            (
                beta_k
                / torch.sqrt(1.0 - alpha_bar_k)
            )
            * pred_noise
        )

        # debug：观察什么时候开始发散
        if debug and k in [99, 95, 90, 80, 50, 20, 0]:
            print(
                f"k = {k:2d} | "
                f"action abs mean = "
                f"{action.abs().mean().item():.6f} | "
                f"pred_noise abs mean = "
                f"{pred_noise.abs().mean().item():.6f} | "
                f"mean abs = "
                f"{mean.abs().mean().item():.6f}"
            )

        # -----------------------------------
        # k = 0 已经是最后一步
        # 不再添加随机噪声
        # -----------------------------------
        if k == 0:
            action = mean
            break

        # -----------------------------------
        # posterior variance
        #
        # beta_tilde =
        # beta_k *
        # (1-alpha_bar_{k-1})
        # /
        # (1-alpha_bar_k)
        # -----------------------------------
        alpha_bar_prev = alpha_bars[k - 1]

        posterior_variance = (
            beta_k
            *
            (1.0 - alpha_bar_prev)
            /
            (1.0 - alpha_bar_k)
        )

        # 防止极小的数值误差导致 sqrt 出问题
        posterior_variance = torch.clamp(
            posterior_variance,
            min=1e-20
        )

        noise = torch.randn_like(action)

        # A^(k-1)
        action = (
            mean
            +
            torch.sqrt(posterior_variance)
            * noise
        )

    return action