import torch.nn as nn
import torch

from models import model
from configs import default as cfg
from datasets.datasets import get_loaders
from utils.train import train_loop
from utils.checkpoint import save_model
from utils.plot import plot_img
from utils.print import print_csv

torch.manual_seed(cfg.seed)

b1 = model.Residual(1, 32, use_1x1conv=True, strides=1)

b2 = nn.Sequential(*model.resnet_block(32, 32, 1, first_block=True))
b3 = nn.Sequential(*model.resnet_block(32, 64, 1))

net = nn.Sequential(b1, b2, b3,
                    nn.AdaptiveAvgPool2d((1,1)),                #What is AdaptiveAvgPool2d?
                    nn.Flatten(), nn.Linear(64, 10)).to(cfg.device)

train_loader,test_loader = get_loaders(cfg.DATA_DIR, cfg.train_batch_size, cfg.test_batch_size)
losses, accs, epochs = train_loop(net, train_loader, test_loader,
                          num_epochs=cfg.num_epochs, lr=cfg.lr, device=cfg.device)

save_model(net)
plot_img(epochs,losses,accs)

num_params = sum(p.numel() for p in net.parameters())

print_csv(epochs,cfg.seed,cfg.lr,losses,accs,num_params)
print(f"Final acc: {accs[-1]:.4f}")