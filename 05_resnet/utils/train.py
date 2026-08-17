import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .evaluate import evaluate
from .checkpoint import save_model
from .plot import plot_img
from .print import print_csv
from models.model import build_net
from datasets.datasets import get_loaders


def set_seed(seed, device):
    """按设备设置随机种子，保证不同 device 上都能复现"""
    torch.manual_seed(seed)               # CPU（模型初始化、DataLoader shuffle）
    if device == "mps":
        torch.mps.manual_seed(seed)       # Apple Silicon
    elif device == "cuda":
        torch.cuda.manual_seed_all(seed)  # NVIDIA GPU


def train_loop(model, train_loader, test_loader, num_epochs, lr, device):
    """纯训练循环：给定模型和数据，返回每个 epoch 的 loss/acc"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epoch_loss = []
    epoch_acc = []
    epoch_num = np.arange(1, num_epochs + 1)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        acc = evaluate(model, test_loader, device)

        print(f"Epoch{epoch + 1}/{num_epochs},"
              f"Loss:{avg_loss:.4f},"
              f"Accuracy:{acc:.4f}")

        epoch_loss.append(avg_loss)
        epoch_acc.append(acc)

    return epoch_loss, epoch_acc, epoch_num


def run_experiment(seed, lr, cfg):
    """一次完整实验：设种子 → 建模型 → 建数据 → 训练 → 保存 → 记录"""
    set_seed(seed, cfg.device)                  # ① 最先设种子

    net = build_net(cfg.device)                 # ② 再建模型
    train_loader, test_loader = get_loaders(    # ③ 再建 DataLoader
        cfg.DATA_DIR, cfg.train_batch_size, cfg.test_batch_size
    )

    losses, accs, epochs = train_loop(
        net, train_loader, test_loader,
        num_epochs=cfg.num_epochs, lr=lr, device=cfg.device,
    )

    num_params = sum(p.numel() for p in net.parameters())

    save_model(net, seed, lr)
    plot_img(epochs, seed, lr, losses, accs, num_params)
    print_csv(epochs, seed, lr, losses, accs, num_params)

    print(f"[seed={seed}, lr={lr}] Final acc: {accs[-1]:.4f}, best acc: {max(accs):.4f}")
    return seed, lr, accs[-1], max(accs)
