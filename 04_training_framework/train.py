import torch
import models.cnn as cnn
import datasets.mnist as mnist
import configs.default as cfg
from utils.train import train_loop
from utils.checkpoint import save_model
from utils.plot import plot_img

net = cnn.CNN().to(cfg.device)
losses, accs, epochs = train_loop(net, mnist.train_loader, mnist.test_loader,
                          num_epochs=cfg.num_epochs, lr=cfg.lr, device=cfg.device)

save_model(net)
plot_img(epochs,losses,accs)

print(f"Final acc: {accs[-1]:.4f}")