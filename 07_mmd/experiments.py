from mmd import mmd_squared
import torch

n_samples=500
sigma=1.0
dim=10

torch.manual_seed(42)

x = torch.randn(n_samples, dim)
y = torch.normal(0, 2, (n_samples, dim))

mmd_result = mmd_squared(x, y, sigma=sigma)
print(f"n_samples={n_samples} MMD²的期望值: {mmd_result.item():.6f}")
