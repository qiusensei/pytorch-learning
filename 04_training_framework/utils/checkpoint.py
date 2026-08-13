import torch
from pathlib import Path

CHECKPOINT_PATH = (
        Path(__file__).resolve().parents[1]
        / "checkpoints"
        / "mnist_cnn.pth")


def save_model(model):
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(),CHECKPOINT_PATH)            #保存模型

def load_model(model, path=CHECKPOINT_PATH, device="cpu"):
    model.load_state_dict(torch.load(path, map_location=device))
    return model