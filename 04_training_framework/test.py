import torch
import models.cnn as cnn
import datasets.mnist as mnist
import configs.default as cfg
from utils.evaluate import evaluate
from utils.checkpoint import load_model

device = cfg.device

net = cnn.CNN().to(device)
net = load_model(net,device)

loader = mnist.get_loaders(cfg.DATA_DIR,cfg.train_batch_size,cfg.test_batch_size)
test_loader = loader[1]

acc = evaluate(net, test_loader, device)

print(f"Test acc: {acc:.4f}")
print("CNN params:", sum(p.numel() for p in net.parameters()))