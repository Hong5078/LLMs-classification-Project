"""RQ1: output-only six-way classification. Validation tuning, then three seeds per model.

Usage: python -m src.exp_rq1
"""
import itertools

import numpy as np
import pandas as pd

from .baseline import run_baseline
from .config import RESULTS, load_config, read_json, write_json
from .data import load_subset, make_data
from .train import run_neural

OUT = RESULTS / "rq1"
TUNE = RESULTS / "tuning"


def tune(cfg, data):
    """Grid search on the validation split only (seed 42). The test split is not evaluated here."""
    rows, best = [], {}
    for arch in ("cnn", "lstm"):
        grid = cfg["tuning"][arch]
        keys = list(grid)
        for values in itertools.product(*(grid[k] for k in keys)):
            arch_cfg = {**cfg[arch], **dict(zip(keys, values))}
            name = f"tune_{arch}_" + "_".join(f"{k}{v}" for k, v in zip(keys, values))
            r = run_neural(name, TUNE, arch, arch_cfg, cfg, data, cfg["models"], cfg["seed"], eval_test=False)
            rows.append({"arch": arch, **dict(zip(keys, values)), "val_accuracy": r["val"]["accuracy"],
                         "val_macro_f1": r["val"]["macro_f1"], "best_epoch": r["best_epoch"],
                         "epochs_run": r["epochs_run"], "sec_per_epoch": r["sec_per_epoch"],
                         "n_params": r["n_params"]})
            if arch not in best or r["val"]["macro_f1"] > best[arch][0]:
                best[arch] = (r["val"]["macro_f1"], arch_cfg)
    pd.DataFrame(rows).to_csv(TUNE / "tuning_summary.csv", index=False)
    write_json({a: c for a, (_, c) in best.items()}, OUT / "best_config.json")
    return {a: c for a, (_, c) in best.items()}


def main():
    cfg = load_config()
    df = load_subset(cfg)
    data = make_data(df, cfg, "output")
    best = tune(cfg, data)

    run_baseline("baseline_output", OUT, cfg, data, cfg["models"], tune_C=True)

    rows = []
    for arch in ("cnn", "lstm"):
        for seed in cfg["seeds_rq1"]:
            r = run_neural(f"{arch}_seed{seed}", OUT, arch, best[arch], cfg, data, cfg["models"], seed,
                           save_preds=(seed == cfg["seed"]))
            rows.append({"arch": arch, "seed": seed, "test_accuracy": r["test"]["accuracy"],
                         "test_macro_f1": r["test"]["macro_f1"], "val_macro_f1": r["val"]["macro_f1"],
                         "best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"],
                         "train_time_sec": r["train_time_sec"],
                         **{f"f1_{m}": r["test"]["per_class_f1"][m] for m in cfg["models"]}})
    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / "runs.csv", index=False)

    num = runs.drop(columns=["seed"]).groupby("arch")
    summary = num.mean().add_suffix("_mean").join(num.std(ddof=1).add_suffix("_std"))
    summary.to_csv(OUT / "summary.csv")
    print(summary[["test_accuracy_mean", "test_accuracy_std", "test_macro_f1_mean", "test_macro_f1_std"]])

    # Mean normalised confusion matrix over seeds, used for the figure.
    for arch in ("cnn", "lstm"):
        cms =[read_json(OUT / f"{arch}_seed{s}.json")["test"]["confusion_matrix_normalized"] for s in cfg["seeds_rq1"]]
        write_json({"classes": cfg["models"], "mean_confusion_matrix_normalized": np.mean(cms, axis=0)},
                   OUT / f"{arch}_mean_confusion.json")


if __name__ == "__main__":
    main()
