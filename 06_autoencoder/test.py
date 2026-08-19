import configs.default as cfg
from utils.evaluate import evaluate
from utils.checkpoint import load_model
import models.autoencoder as autoencoder

from datasets.datasets import get_loaders

seed = cfg.seeds[0]
lr = cfg.lrs[0]

net = autoencoder.SimpleAutoencoder().to(cfg.device)
net = load_model(net, cfg.device, seed=seed, lr=lr)

_, test_loader = get_loaders(cfg.DATA_DIR, cfg.train_batch_size, cfg.test_batch_size)
loss = evaluate(net, test_loader, cfg.device)

print(f"Test loss: {loss:.4f}")
print("Params:", sum(p.numel() for p in net.parameters()))
