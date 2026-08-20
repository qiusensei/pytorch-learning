import torch

def gaussian_kernel(x, y, sigma=1.0, eps=1e-8):
    sigma_safe = max(sigma,eps)
    dist_sq = torch.cdist(x, y, p=2) ** 2
    return torch.exp(-dist_sq / (2 * (sigma_safe ** 2)))

def mmd_squared(x, y, sigma):
    m = x.size(0)
    n = y.size(0)

    k_xx = gaussian_kernel(x, x, sigma=sigma)
    k_yy = gaussian_kernel(y, y, sigma=sigma)
    k_xy = gaussian_kernel(x, y, sigma=sigma)

    mask_xx = ~torch.eye(m, dtype=torch.bool)
    mask_yy = ~torch.eye(n, dtype=torch.bool)

    loss_xx = k_xx[mask_xx].sum() / (m * (m - 1))
    loss_yy = k_yy[mask_yy].sum() / (n * (n - 1))
    loss_xy = k_xy.sum() / (m * n)

    return loss_xx + loss_yy - 2 * loss_xy