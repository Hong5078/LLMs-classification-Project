"""RQ4 ablations: retrain the output-only models after removing one kind of signal from the responses.

Usage: python -m src.exp_rq4_ablations [--archs cnn tfidf_lr lstm]   (run after src.exp_rq1)
"""
import argparse

import pandas as pd

from .baseline import run_baseline
from .config import RESULTS, best_arch_config, load_config, read_json
from .data import load_subset, make_data
from .tokenizer import tokenize
from .train import run_neural
from .transforms import TRANSFORMS

OUT = RESULTS / "rq4" / "ablations"


def variants(cfg):
    v = {
        "strip_markdown": {"transform": "strip_markdown"},
        "normalize_whitespace": {"transform": "normalize_whitespace"},
        "lower_nopunct": {"transform": "lower_nopunct"},
        "mask_self_id": {"transform": "mask_self_id"},
        "normalize_typography": {"transform": "normalize_typography"},
    }
    for n in cfg["ablation"]["truncate_tokens"]:
        v[f"truncate_{n}"] = {"truncate": n}
    n = min(cfg["ablation"]["truncate_tokens"])
    v[f"lower_nopunct_truncate_{n}"] = {"transform": "lower_nopunct", "truncate": n}
    return v


def share_changed(df, spec):
    """Share of responses that the ablation actually alters."""
    fn = TRANSFORMS[spec.get("transform", "none")]
    changed = df["response"].map(lambda t: fn(t) != t)
    if "truncate" in spec:
        changed |= df["response"].map(lambda t: len(tokenize(fn(t))) > spec["truncate"])
    return float(changed.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archs", nargs="+", default=["cnn", "tfidf_lr", "lstm"])
    args = parser.parse_args()

    cfg = load_config()
    models = cfg["models"]
    df = load_subset(cfg)
    seed = cfg["seed"]
    specs = variants(cfg)
    changed = {name: share_changed(df, spec) for name, spec in specs.items()}

    for arch in args.archs:
        for name, spec in specs.items():
            data = make_data(df, cfg, "output", **spec)
            meta = {"variant": name, **spec}
            if arch == "tfidf_lr":
                run_baseline(f"{arch}_{name}", OUT, cfg, data, models, meta=meta)
            else:
                run_neural(f"{arch}_{name}", OUT, arch, best_arch_config(cfg, arch), cfg, data, models, seed, meta=meta)

    # Summary over whatever has been run so far (the reference is the unmodified seed-42 RQ1 model).
    refs = {"cnn": f"cnn_seed{seed}", "lstm": f"lstm_seed{seed}", "tfidf_lr": "baseline_output"}
    rows = []
    for arch, ref_name in refs.items():
        ref = read_json(RESULTS / "rq1" / f"{ref_name}.json")["test"]
        rows.append({"arch": arch, "variant": "none", "share_changed": 0.0, "test_accuracy": ref["accuracy"],
                     "test_macro_f1": ref["macro_f1"], "accuracy_drop": 0.0, "macro_f1_drop": 0.0,
                     **{f"f1_{m}": ref["per_class_f1"][m] for m in models}})
        for name in specs:
            path = OUT / f"{arch}_{name}.json"
            if not path.exists():
                continue
            t = read_json(path)["test"]
            rows.append({"arch": arch, "variant": name, "share_changed": changed[name],
                         "test_accuracy": t["accuracy"], "test_macro_f1": t["macro_f1"],
                         "accuracy_drop": ref["accuracy"] - t["accuracy"],
                         "macro_f1_drop": ref["macro_f1"] - t["macro_f1"],
                         **{f"f1_{m}": t["per_class_f1"][m] for m in models}})
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "rq4" / "ablations.csv", index=False)
    print(summary[["arch", "variant", "share_changed", "test_accuracy", "accuracy_drop"]])


if __name__ == "__main__":
    main()
