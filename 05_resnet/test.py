import torch.nn as nn
import datasets.datasets as mnist
import configs.default as cfg
from utils.evaluate import evaluate
from models import model
from utils.checkpoint import load_model

device = cfg.device

b1 = model.Residual(1, 32, use_1x1conv=True, strides=1)

b2 = nn.Sequential(*model.resnet_block(32, 32, 1, first_block=True))
b3 = nn.Sequential(*model.resnet_block(32, 64, 1))

net = nn.Sequential(b1, b2, b3,
                    nn.AdaptiveAvgPool2d((1,1)),                #What is AdaptiveAvgPool2d?
                    nn.Flatten(), nn.Linear(64, 10)).to(cfg.device)
net = load_model(net,device)

loader = mnist.get_loaders(cfg.DATA_DIR,cfg.train_batch_size,cfg.test_batch_size)
test_loader = loader[1]

acc = evaluate(net, test_loader, device)

print(f"Test acc: {acc:.4f}")
print("Params:", sum(p.numel() for p in net.parameters()))