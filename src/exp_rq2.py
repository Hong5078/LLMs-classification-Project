"""RQ2: input-only vs output-only vs input+output, for the CNN, the BiLSTM and the TF-IDF baseline.

Usage: python -m src.exp_rq2   (run after src.exp_rq1; the output-only numbers are the seed-42 RQ1 runs)
"""
import pandas as pd

from .baseline import run_baseline
from .config import RESULTS, best_arch_config, load_config, read_json
from .data import load_subset, make_data
from .train import run_neural

OUT = RESULTS / "rq2"


def row(arch, condition, r):
    return {"arch": arch, "condition": condition, "test_accuracy": r["test"]["accuracy"],
            "test_macro_f1": r["test"]["macro_f1"], "val_macro_f1": r["val"]["macro_f1"],
            **{f"f1_{m}": v for m, v in r["test"]["per_class_f1"].items()}}


def main():
    cfg = load_config()
    df = load_subset(cfg)
    seed = cfg["seed"]
    rows = []
    for condition in ("input", "input+output"):
        data = make_data(df, cfg, condition)
        tag = condition.replace("+", "_")
        for arch in ("cnn", "lstm"):
            r = run_neural(f"{arch}_{tag}", OUT, arch, best_arch_config(cfg, arch), cfg, data, cfg["models"], seed,
                           meta={"condition": condition})
            rows.append(row(arch, condition, r))
        rows.append(row("tfidf_lr", condition,
                        run_baseline(f"baseline_{tag}", OUT, cfg, data, cfg["models"], meta={"condition": condition})))

    rq1 = RESULTS / "rq1"
    for arch, name in (("cnn", f"cnn_seed{seed}"), ("lstm", f"lstm_seed{seed}"), ("tfidf_lr", "baseline_output")):
        rows.append(row(arch, "output", read_json(rq1 / f"{name}.json")))

    summary = pd.DataFrame(rows)
    summary["condition"] = pd.Categorical(summary["condition"], ["input", "output", "input+output"])
    summary = summary.sort_values(["arch", "condition"])
    summary.to_csv(OUT / "summary.csv", index=False)
    print(summary[["arch", "condition", "test_accuracy", "test_macro_f1"]])


if __name__ == "__main__":
    main()
