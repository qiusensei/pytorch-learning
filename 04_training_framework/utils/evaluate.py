import torch

def evaluate(model, test_loader, device):
    model.eval()

    total_correct = 0
    total_samples = 0

    with torch.no_grad():  # close grad calculation to save storage
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # ← 加这一行
            outputs = model(images)
            preds = outputs.argmax(dim=1)  # This the result of prediction
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)  # The quantity of samples

    accuracy = total_correct / total_samples

    return accuracy