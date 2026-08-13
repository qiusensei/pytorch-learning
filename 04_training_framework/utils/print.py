import pandas as pd

def print_csv(epoch_num, seed, epoch_loss, epoch_acc):
    pd.DataFrame({"epoch": epoch_num,
                  "seed": seed,
                  "loss": epoch_loss,
                  "acc": epoch_acc}).to_csv("results.csv", index=False)