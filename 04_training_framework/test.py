import torch
import models.cnn as cnn
import datasets.mnist as mnist
from utils.evaluate import evaluate
from utils.checkpoint import load_model

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
net = cnn.CNN().to(device)
net = load_model(net)
acc = evaluate(net, mnist.test_loader, device)
print(f"Test acc: {acc:.4f}")