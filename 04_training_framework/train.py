import torch
import models.cnn as cnn
import datasets.mnist as mnist
import configs.default as cfg
from utils.train import train_loop
from utils.checkpoint import save_model
from utils.plot import plot_img

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
net = cnn.CNN().to(device)
losses, accs, epochs = train_loop(net, mnist.train_loader, mnist.test_loader,
                          num_epochs=cfg.num_epochs, lr=cfg.lr, device=device)

save_model(net)
plot_img(epochs,losses,accs)

print(f"Final acc: {accs[-1]:.4f}")