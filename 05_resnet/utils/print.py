import pandas as pd
from pathlib import Path

def print_csv(epoch_num, seed, lr, epoch_loss, epoch_acc, num_params):
    framework_dir = Path(__file__).resolve().parent.parent
    results_dir = framework_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / f"train_log_seed{seed}_lr{lr}_params{num_params}.csv"

    pd.DataFrame({"epoch": epoch_num,
                  "seed": seed,
                  "lr": lr,
                  "loss": epoch_loss,
                  "acc": epoch_acc,
                  "num_params": num_params}).to_csv(results_path, index=False)