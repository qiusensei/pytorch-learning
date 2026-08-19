import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import models.autoencoder as autoencoder

from .evaluate import evaluate
from .checkpoint import save_model
from .plot import plot_img
from .print import print_csv
from datasets.datasets import get_loaders


def set_seed(seed, device):
    """按设备设置随机种子，保证不同 device 上都能复现"""
    torch.manual_seed(seed)               # CPU（模型初始化、DataLoader shuffle）
    if device == "mps":
        torch.mps.manual_seed(seed)       # Apple Silicon
    elif device == "cuda":
        torch.cuda.manual_seed_all(seed)  # NVIDIA GPU


def train_loop(model, train_loader, test_loader, num_epochs, lr, device):
    """纯训练循环：给定模型和数据，返回每个 epoch 的 train_loss 和 test_loss"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_losses = []
    epoch_num = np.arange(1, num_epochs + 1)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        test_loss = evaluate(model, test_loader, device)

        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"train_loss: {train_loss:.4f}, "
              f"test_loss: {test_loss:.4f}")

        train_losses.append(train_loss)
        test_losses.append(test_loss)

    return train_losses, test_losses, epoch_num


def run_experiment(seed, lr, cfg):
    """一次完整实验：设种子 → 建模型 → 建数据 → 训练 → 保存 → 记录"""
    set_seed(seed, cfg.device)                  # ① 最先设种子

    net = autoencoder.SimpleAutoencoder().to(cfg.device)           # ② 再建模型
    train_loader, test_loader = get_loaders(    # ③ 再建 DataLoader
        cfg.DATA_DIR, cfg.train_batch_size, cfg.test_batch_size
    )

    train_losses, test_losses, epochs = train_loop(
        net, train_loader, test_loader,
        num_epochs=cfg.num_epochs, lr=lr, device=cfg.device,
    )

    num_params = sum(p.numel() for p in net.parameters())

    save_model(net, seed, lr)
    plot_img(epochs, seed, lr, train_losses, test_losses, num_params)
    print_csv(epochs, seed, lr, train_losses, test_losses, num_params)

    print(f"[seed={seed}, lr={lr}, last_train_loss={train_losses[-1]:.4f}, last_test_loss={test_losses[-1]:.4f}]")
    return seed, lr, train_losses[-1], test_losses[-1]
