import torch
import models.cnn as cnn
import configs.default as cfg
from datasets.mnist import get_loaders
from utils.train import train_loop
from utils.checkpoint import save_model
from utils.plot import plot_img

torch.manual_seed(cfg.seed)

net = cnn.CNN().to(cfg.device)
train_loader,test_loader = get_loaders(cfg.DATA_DIR, cfg.train_batch_size, cfg.test_batch_size)
losses, accs, epochs = train_loop(net, train_loader, test_loader,
                          num_epochs=cfg.num_epochs, lr=cfg.lr, device=cfg.device)

save_model(net)
plot_img(epochs,losses,accs)

print(f"Final acc: {accs[-1]:.4f}")