import torch
import torch.nn as nn

def evaluate(model, test_loader, device):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, _ in test_loader:      # 注意：labels 用 _ 丢弃
            images = images.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, images).item()
            num_batches += 1

    return total_loss / num_batches        # 返回平均重建损失