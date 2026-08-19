import matplotlib.pyplot as plt
from pathlib import Path

PLOT_DIR = Path(__file__).resolve().parents[1] / "plots"

def plot_img(epoch_num, seed, lr, train_loss, test_loss, num_params):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    loss_path = PLOT_DIR / f"loss_curve_autoencoder_seed{seed}_lr{lr}_params{num_params}.png"

    plt.figure()
    plt.plot(epoch_num, train_loss, label="Train Loss", marker="o")
    plt.plot(epoch_num, test_loss, label="Test Loss", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Reconstruction Loss Curve")
    plt.legend()
    plt.savefig(loss_path, dpi=150)