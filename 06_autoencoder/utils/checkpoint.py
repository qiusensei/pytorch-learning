import torch
from pathlib import Path

CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"


def save_model(model, seed, lr):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"mnist_autoencoder_seed{seed}_lr{lr}.pth"
    torch.save(model.state_dict(), path)
    return path


def load_model(model, device, seed, lr):
    path = CHECKPOINT_DIR / f"mnist_autoencoder_seed{seed}_lr{lr}.pth"
    model.load_state_dict(torch.load(path, map_location=device))
    return model
