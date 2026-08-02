import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import dataset
import model
import os

import matplotlib.pyplot as plt

os.makedirs("checkpoints", exist_ok=True)       #Create a folder to save my model.
os.makedirs("plots", exist_ok=True)             #Create a folder to save train result images

num_epochs = 10
lr = 1e-3

if __name__ == "__main__":
    net = model.MyMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    epoch_loss = []
    epoch_acc = []
    epoch_num = np.arange(1, num_epochs + 1)

    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        for images, labels in dataset.train_loader:
            outputs = net(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(dataset.train_loader)
        epoch_loss.append(avg_loss)
        print(f"Epoch{epoch + 1}/{num_epochs},"
              f"Loss:{avg_loss:.4f}")

        net.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():  # close grad calculation to save storage
            for images, labels in dataset.test_loader:
                outputs = net(images)
                preds = outputs.argmax(dim=1)  # This the result of prediction
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)  # The quantity of samples
        accuracy = total_correct / total_samples
        epoch_acc.append(accuracy)
        print(f"Test Accuracy:{accuracy:.4f}")

    torch.save(net.state_dict(),"checkpoints/mnist_mlp.pth")            #保存模型

    plt.figure()
    plt.plot(epoch_num,epoch_loss,label = "Train Loss", marker = "o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.savefig("plots/loss_curve.png",dpi=150)

    plt.figure()
    plt.plot(epoch_num,epoch_acc,label = "Test Acc", marker = "o")
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.title("Test Acc Curve")
    plt.legend()
    plt.savefig("plots/accuracy_curve.png",dpi=150)