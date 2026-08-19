from configs import default as cfg
from utils.train import run_experiment


if __name__ == "__main__":
    summary = []
    for seed in cfg.seeds:
        for lr in cfg.lrs:
            print(f"\n{'=' * 50}\nRunning: seed={seed}, lr={lr}\n{'=' * 50}")
            summary.append(run_experiment(seed, lr, cfg))

    print(f"\n{'=' * 50}\nSummary\n{'=' * 50}")
    print(f"{'seed':>6} {'lr':>9} {'train_loss':>12} {'test_loss':>12}")
    for seed, lr, train_loss, test_loss in summary:
        print(f"{seed:>6} {lr:>9} {train_loss:>12.4f} {test_loss:>12.4f}")
