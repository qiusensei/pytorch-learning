import configs.default as cfg
from utils.evaluate import evaluate
from utils.checkpoint import load_model
from models.model import build_net
from datasets.datasets import get_loaders

seed = cfg.seeds[0]
lr = cfg.lrs[0]

net = build_net(cfg.device)
net = load_model(net, cfg.device, seed=seed, lr=lr)

_, test_loader = get_loaders(cfg.DATA_DIR, cfg.train_batch_size, cfg.test_batch_size)
acc = evaluate(net, test_loader, cfg.device)

print(f"Test acc: {acc:.4f}")
print("Params:", sum(p.numel() for p in net.parameters()))
