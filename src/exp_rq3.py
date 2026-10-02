"""RQ3: do fingerprints transfer across question types?

TXD-22 has no coding/maths/writing tasks, so the "domain" is the form of the question:
definitional ("What is ...", "Define ...") versus explanatory ("Explain ...", "How ...", "Discuss ...").
Each model is trained on one type and tested on its own type (in-domain) and on the other (cross-domain).
The larger definitional training set is also subsampled to the size of the explanatory one so that the
in-domain and cross-domain models being compared saw the same number of prompts.

Usage: python -m src.exp_rq3   (run after src.exp_rq1)
"""
import numpy as np
import pandas as pd

from .baseline import run_baseline
from .config import RESULTS, best_arch_config, load_config, output_max_len
from .data import build_inputs, by_split, load_subset
from .train import run_neural

OUT = RESULTS / "rq3"
DOMAINS = ("definitional", "explanatory")
ARCHS = ("cnn", "lstm", "tfidf_lr")


def main():
    cfg = load_config()
    models = cfg["models"]
    df = load_subset(cfg)
    tokens, strings = build_inputs(df, cfg, "output", out_len=output_max_len(cfg))
    in_domain = {d: (df["qtype"] == d).to_numpy() for d in DOMAINS}
    is_train = (df["split"] == "train").to_numpy()

    # Size-matched definitional training set: as many prompt groups as the explanatory training set has.
    rng = np.random.default_rng(cfg["seed"])
    exp_groups = np.sort(df.loc[is_train & in_domain["explanatory"], "group"].unique())
    def_groups = np.sort(df.loc[is_train & in_domain["definitional"], "group"].unique())
    keep = set(rng.choice(def_groups, size=len(exp_groups), replace=False))
    matched = in_domain["definitional"] & (~is_train | df["group"].isin(keep).to_numpy())

    training_sets = {
        "definitional_matched": (matched, "definitional"),
        "explanatory": (in_domain["explanatory"], "explanatory"),
        "definitional_full": (in_domain["definitional"], "definitional"),
    }

    rows = []
    for train_name, (mask, source) in training_sets.items():
        target = DOMAINS[1 - DOMAINS.index(source)]
        data = by_split(df, tokens, strings, mask)            # train, val and in-domain test: source type only
        cross = {"cross_test": by_split(df, tokens, strings, in_domain[target])["test"]}
        meta = {"train_set": train_name, "source": source, "target": target}
        for arch in ARCHS:
            name = f"{arch}_train-{train_name}"
            if arch == "tfidf_lr":
                r = run_baseline(name, OUT, cfg, data, models, extra_eval=cross, meta=meta)
            else:
                r = run_neural(name, OUT, arch, best_arch_config(cfg, arch), cfg, data, models, cfg["seed"],
                               extra_eval=cross, meta=meta)
            rows.append({"arch": arch, **meta, "n_train": r["n_train"],
                         "n_in_test": r["test"]["n"], "n_cross_test": r["cross_test"]["n"],
                         "in_accuracy": r["test"]["accuracy"], "in_macro_f1": r["test"]["macro_f1"],
                         "cross_accuracy": r["cross_test"]["accuracy"], "cross_macro_f1": r["cross_test"]["macro_f1"],
                         **{f"in_f1_{m}": r["test"]["per_class_f1"][m] for m in models},
                         **{f"cross_f1_{m}": r["cross_test"]["per_class_f1"][m] for m in models}})
    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / "runs.csv", index=False)

    # For each test domain, compare the model trained on that domain with the size-matched model
    # trained on the other domain. Both are evaluated on the same test rows.
    matched_for = {"definitional": "definitional_matched", "explanatory": "explanatory"}
    transfer = []
    for arch in ARCHS:
        sub = runs[runs["arch"] == arch].set_index("train_set")
        for test_domain in DOMAINS:
            other = DOMAINS[1 - DOMAINS.index(test_domain)]
            ind, cross = sub.loc[matched_for[test_domain]], sub.loc[matched_for[other]]
            row = {"arch": arch, "test_domain": test_domain, "n_test": ind["n_in_test"],
                   "in_train_set": matched_for[test_domain], "cross_train_set": matched_for[other],
                   "in_accuracy": ind["in_accuracy"], "cross_accuracy": cross["cross_accuracy"],
                   "accuracy_drop": ind["in_accuracy"] - cross["cross_accuracy"],
                   "in_macro_f1": ind["in_macro_f1"], "cross_macro_f1": cross["cross_macro_f1"],
                   "macro_f1_drop": ind["in_macro_f1"] - cross["cross_macro_f1"]}
            for m in models:
                row[f"in_f1_{m}"] = ind[f"in_f1_{m}"]
                row[f"cross_f1_{m}"] = cross[f"cross_f1_{m}"]
                row[f"f1_drop_{m}"] = ind[f"in_f1_{m}"] - cross[f"cross_f1_{m}"]
            transfer.append(row)
    transfer = pd.DataFrame(transfer)
    transfer.to_csv(OUT / "transfer.csv", index=False)
    print(transfer[["arch", "test_domain", "in_accuracy", "cross_accuracy", "accuracy_drop"]])


if __name__ == "__main__":
    main()
