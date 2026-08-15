from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
FRAMEWORK_DIR = BASE_DIR.parent
REPO_DIR = FRAMEWORK_DIR.parent
DATA_DIR = REPO_DIR / "data"

num_epochs = 10
lr = 1e-3
train_batch_size = 64
test_batch_size = 1000
seed = 42
device = "mps"