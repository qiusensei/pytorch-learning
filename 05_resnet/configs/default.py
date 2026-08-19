from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent
FRAMEWORK_DIR = BASE_DIR.parent
REPO_DIR = FRAMEWORK_DIR.parent
DATA_DIR = REPO_DIR / "data"

num_epochs = 10

# 一次跑完的实验组合
seeds = [42, 919, 1000]
lrs = [5e-4]

train_batch_size = 64
test_batch_size = 1000

def _mps_available():
    return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()


def pick_device(preferred=None):
    """选择设备。默认自动检测（cuda > mps > cpu）；也可传 "cuda"/"mps"/"cpu" 强制指定。"""
    if preferred is not None:
        if preferred == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device=cuda 但当前机器没有可用的 CUDA")
        if preferred == "mps" and not _mps_available():
            raise RuntimeError("device=mps 但当前机器不支持 MPS（仅 macOS/Apple Silicon 可用）")
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if _mps_available():
        return "mps"
    return "cpu"


# 默认自动检测设备；想强制指定可改成 pick_device("cuda") / pick_device("mps") / pick_device("cpu")
device = pick_device()
