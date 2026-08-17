from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
FRAMEWORK_DIR = BASE_DIR.parent
REPO_DIR = FRAMEWORK_DIR.parent
DATA_DIR = REPO_DIR / "data"

num_epochs = 10

# 一次跑完的实验组合：3 个 seed × 3 个 lr
seeds = [42, 919, 1000]
lrs = [3e-4, 4e-4, 5e-4]

train_batch_size = 64
test_batch_size = 1000

# device 可切换："mps"（Apple Silicon）/ "cuda"（NVIDIA GPU）/ "cpu"
device = "mps"
