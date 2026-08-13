import matplotlib.pyplot as plt
import os

def plot_img(epoch_num, epoch_loss, epoch_acc):
    os.makedirs("plots", exist_ok=True)

    plt.figure()
    plt.plot(epoch_num, epoch_loss, label="Train Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.savefig("plots/loss_curve_cnn.png", dpi=150)

    plt.figure()
    plt.plot(epoch_num, epoch_acc, label="Test Acc", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.title("Test Acc Curve")
    plt.legend()
    plt.savefig("plots/accuracy_curve_cnn.png", dpi=150)
