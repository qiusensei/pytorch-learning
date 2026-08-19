from torch.utils.data import Dataset, DataLoader
import torchvision


train_dataset = torchvision.datasets.MNIST('../../data',
                                           train=True,
                                           transform=torchvision.transforms.ToTensor(),
                                           download=True)
test_dataset  = torchvision.datasets.MNIST('../../data',
                                           train=False,
                                           transform=torchvision.transforms.ToTensor(),
                                           download=True)
train_loader = DataLoader(train_dataset,
                          batch_size=64,
                          shuffle=True)
test_loader = DataLoader(test_dataset,
                         batch_size=1000,
                         shuffle=False)

if __name__ == "__main__":
    print(len(train_dataset),len(test_dataset))
    print(len(train_loader),len(test_loader))

    train_features,train_labels = next(iter(train_loader))
    print(f"训练 batch: features {train_features.shape}, labels {train_labels.shape}")

    test_features, test_labels = next(iter(test_loader))
    print(f"测试 batch: features {test_features.shape}, labels {test_labels.shape}")
