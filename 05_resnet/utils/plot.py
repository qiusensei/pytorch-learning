import matplotlib.pyplot as plt
from pathlib import Path

PLOT_DIR = Path(__file__).resolve().parents[1] / "plots"

def plot_img(epoch_num, epoch_loss, epoch_acc):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    loss_path = PLOT_DIR / "loss_curve_resnet.png"
    acc_path = PLOT_DIR / "accuracy_curve_resnet.png"

    plt.figure()
    plt.plot(epoch_num, epoch_loss, label="Train Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.savefig(loss_path, dpi=150)

    plt.figure()
    plt.plot(epoch_num, epoch_acc, label="Test Acc", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.title("Test Acc Curve")
    plt.legend()
    plt.savefig(acc_path, dpi=150)
