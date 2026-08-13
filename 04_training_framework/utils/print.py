import pandas as pd
from pathlib import Path

def print_csv(epoch_num, seed, epoch_loss, epoch_acc):
    framework_dir = Path(__file__).resolve().parent.parent
    results_dir = framework_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / "train_log.csv"

    pd.DataFrame({"epoch": epoch_num,
                  "seed": seed,
                  "loss": epoch_loss,
                  "acc": epoch_acc}).to_csv(results_path, index=False)