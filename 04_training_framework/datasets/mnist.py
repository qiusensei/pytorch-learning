from torch.utils.data import DataLoader
import torchvision
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
FRAMEWORK_DIR = BASE_DIR.parent
REPO_DIR = FRAMEWORK_DIR.parent
DATA_DIR = REPO_DIR / "data"

train_dataset = torchvision.datasets.MNIST(root=DATA_DIR,
                                           train=True,
                                           transform=torchvision.transforms.ToTensor(),
                                           download=True)
test_dataset  = torchvision.datasets.MNIST(root=DATA_DIR,
                                           train=False,
                                           transform=torchvision.transforms.ToTensor(),
                                           download=True)
train_loader = DataLoader(train_dataset,
                          batch_size=64,
                          shuffle=True)
test_loader = DataLoader(test_dataset,
                         batch_size=1000,
                         shuffle=False)

