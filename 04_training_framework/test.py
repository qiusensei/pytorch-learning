import torch
import models.cnn as cnn
import datasets.mnist as mnist
import configs.default as cfg
from utils.evaluate import evaluate
from utils.checkpoint import load_model


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
net = cnn.CNN().to(device)
net = load_model(net)
loader = mnist.get_loaders(cfg.DATA_DIR,cfg.train_batch_size,cfg.test_batch_size)
test_loader = loader[1]
acc = evaluate(net, test_loader, device)

print(f"Test acc: {acc:.4f}")