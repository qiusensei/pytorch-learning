from torch.utils.data import DataLoader
import torchvision

def get_loaders(data_dir, train_batch_size, test_batch_size):
    train_dataset = torchvision.datasets.MNIST(root=data_dir,
                                               train=True,
                                               transform=torchvision.transforms.ToTensor(),
                                               download=True)
    test_dataset = torchvision.datasets.MNIST(root=data_dir,
                                              train=False,
                                              transform=torchvision.transforms.ToTensor(),
                                              download=True)
    train_loader = DataLoader(train_dataset,
                              train_batch_size,
                              shuffle=True)
    test_loader = DataLoader(test_dataset,
                             test_batch_size,
                             shuffle=False)

    return train_loader,test_loader