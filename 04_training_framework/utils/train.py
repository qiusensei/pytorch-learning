import numpy as np
import torch.nn as nn
import torch.optim as optim
from .evaluate import evaluate

def train_loop(model, train_loader, test_loader, num_epochs, lr, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epoch_loss = []
    epoch_acc = []
    epoch_num = np.arange(1, num_epochs+1)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # ← 加这一行
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        acc =  evaluate(model,test_loader,device)

        print(f"Epoch{epoch + 1}/{num_epochs},"
              f"Loss:{avg_loss:.4f},"
              f"Accuracy:{acc:.4f}")

        epoch_loss.append(avg_loss)
        epoch_acc.append(acc)



    return epoch_loss,epoch_acc,epoch_num