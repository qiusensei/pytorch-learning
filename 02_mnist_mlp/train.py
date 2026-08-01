import torch
import torch.nn as nn
import torch.optim as optim
import dataset
import model
import os

os.makedirs("checkpoints", exist_ok=True)       #Create a dir to save my model.

num_epochs = 5
lr = 1e-3

if __name__ == "__main__":
    net = model.MyMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        for images, labels in dataset.train_loader:
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()
        print(f"Epoch{epoch + 1}/{num_epochs},"
              f"Loss:{running_loss / len(dataset.train_loader):.4f}")
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
        print(f"Test Accuracy:{accuracy:.4f}")

    torch.save(net.state_dict(),"checkpoints/mnist_mlp.pth")