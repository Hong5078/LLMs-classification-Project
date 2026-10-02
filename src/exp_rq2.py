"""RQ2: input-only vs output-only vs input+output, for the CNN, the BiLSTM and the TF-IDF baseline.

Also trains input-only models on ASCII-normalised prompts and breaks input-only accuracy down by
whether the exact prompt string is shared by all six models, to explain any above-chance result.

Usage: python -m src.exp_rq2   (run after src.exp_rq1; the output-only numbers are the seed-42 RQ1 runs)
"""
import pandas as pd

from .baseline import run_baseline
from .config import RESULTS, best_arch_config, load_config, read_json, write_json
from .data import load_subset, make_data
from .train import run_neural

OUT = RESULTS / "rq2"
ARCHS = ("cnn", "lstm", "tfidf_lr")


def row(arch, condition, r):
    return {"arch": arch, "condition": condition, "test_accuracy": r["test"]["accuracy"],
            "test_macro_f1": r["test"]["macro_f1"], "val_macro_f1": r["val"]["macro_f1"],
            **{f"f1_{m}": v for m, v in r["test"]["per_class_f1"].items()}}


def input_only_breakdown(df, models):
    """Input-only test accuracy on prompts whose exact string all six models share, and on the rest."""
    n_models = df.groupby("prompt")["model"].transform("nunique")
    shared = dict(zip(df["row_id"], n_models == len(models)))
    out = {}
    for arch in ARCHS:
        name = "baseline_input" if arch == "tfidf_lr" else f"{arch}_input"
        p = pd.read_csv(OUT / f"{name}_test_preds.csv")
        p["shared"] = p["row_id"].map(shared)
        p["correct"] = p["true"] == p["pred"]
        not_shared = p[~p["shared"]]
        out[arch] = {
            "n_shared": int(p["shared"].sum()),
            "accuracy_shared": float(p.loc[p["shared"], "correct"].mean()),
            "n_not_shared": int(len(not_shared)),
            "accuracy_not_shared": float(not_shared["correct"].mean()),
            "not_shared_by_true_model": not_shared.groupby("true").size().to_dict(),
            "not_shared_correct_by_true_model": not_shared[not_shared["correct"]].groupby("true").size().to_dict(),
            "predicted_counts": p["pred"].value_counts().to_dict(),
        }
    write_json(out, OUT / "input_only_breakdown.json")
    return out


def main():
    cfg = load_config()
    models = cfg["models"]
    df = load_subset(cfg)
    seed = cfg["seed"]
    rows = []
    for condition in ("input", "input_ascii", "input+output"):
        data = make_data(df, cfg, condition)
        tag = condition.replace("+", "_")
        for arch in ("cnn", "lstm"):
            r = run_neural(f"{arch}_{tag}", OUT, arch, best_arch_config(cfg, arch), cfg, data, models, seed,
                           meta={"condition": condition}, save_preds=True)
            rows.append(row(arch, condition, r))
        rows.append(row("tfidf_lr", condition,
                        run_baseline(f"baseline_{tag}", OUT, cfg, data, models, meta={"condition": condition},
                                     save_preds=True)))

    rq1 = RESULTS / "rq1"
    for arch, name in (("cnn", f"cnn_seed{seed}"), ("lstm", f"lstm_seed{seed}"), ("tfidf_lr", "baseline_output")):
        rows.append(row(arch, "output", read_json(rq1 / f"{name}.json")))

    summary = pd.DataFrame(rows)
    summary["condition"] = pd.Categorical(summary["condition"], ["input", "input_ascii", "output", "input+output"])
    summary = summary.sort_values(["arch", "condition"])
    summary.to_csv(OUT / "summary.csv", index=False)
    print(summary[["arch", "condition", "test_accuracy", "test_macro_f1"]])
    print(input_only_breakdown(df, models))


if __name__ == "__main__":
    main()
