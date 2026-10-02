"""Error analysis of the seed-42 RQ1 models on the test split.

Usage: python -m src.error_analysis   (run after src.exp_rq1)
"""
import pandas as pd

from .config import RESULTS, load_config, write_json
from .data import load_subset
from .tokenizer import tokenize

OUT = RESULTS / "rq1"


def main():
    cfg = load_config()
    seed = cfg["seed"]
    df = load_subset(cfg).set_index("row_id")
    cnn = pd.read_csv(OUT / f"cnn_seed{seed}_test_preds.csv").set_index("row_id")
    lstm = pd.read_csv(OUT / f"lstm_seed{seed}_test_preds.csv").set_index("row_id")

    t = pd.DataFrame({"true": cnn["true"], "cnn": cnn["pred"], "lstm": lstm["pred"]})
    t["n_tokens"] = df.loc[t.index, "response"].map(lambda s: len(tokenize(s)))
    t["qtype"] = df.loc[t.index, "qtype"]
    cnn_wrong, lstm_wrong = t["cnn"] != t["true"], t["lstm"] != t["true"]
    both_wrong = cnn_wrong & lstm_wrong

    pairs = (t[cnn_wrong].groupby(["true", "cnn"]).size().rename("cnn_errors").to_frame()
             .join(t[lstm_wrong].groupby(["true", "lstm"]).size().rename("lstm_errors"), how="outer")
             .fillna(0).astype(int))
    pairs.index.names = ["true", "predicted"]
    pairs["total"] = pairs.sum(axis=1)
    pairs = pairs.sort_values("total", ascending=False)
    pairs.to_csv(OUT / "error_pairs.csv")

    stats = {
        "n_test": int(len(t)),
        "cnn_errors": int(cnn_wrong.sum()), "lstm_errors": int(lstm_wrong.sum()),
        "both_wrong": int(both_wrong.sum()),
        "both_wrong_same_prediction": int((both_wrong & (t["cnn"] == t["lstm"])).sum()),
        "either_wrong": int((cnn_wrong | lstm_wrong).sum()),
        "mean_tokens_cnn_correct": float(t.loc[~cnn_wrong, "n_tokens"].mean()),
        "mean_tokens_cnn_wrong": float(t.loc[cnn_wrong, "n_tokens"].mean()),
        "mean_tokens_lstm_correct": float(t.loc[~lstm_wrong, "n_tokens"].mean()),
        "mean_tokens_lstm_wrong": float(t.loc[lstm_wrong, "n_tokens"].mean()),
        "cnn_error_rate_by_true_model": cnn_wrong.groupby(t["true"]).mean().to_dict(),
        "lstm_error_rate_by_true_model": lstm_wrong.groupby(t["true"]).mean().to_dict(),
        "cnn_error_rate_by_qtype": cnn_wrong.groupby(t["qtype"]).mean().to_dict(),
        "lstm_error_rate_by_qtype": lstm_wrong.groupby(t["qtype"]).mean().to_dict(),
        "top_pairs": [{"true": a, "predicted": b, **row} for (a, b), row in pairs.head(6).astype(int).iterrows()],
    }
    write_json(stats, OUT / "error_stats.json")

    # One example per frequent confusion: the first test row (lowest row_id) on which both models
    # made that same mistake. Chosen by rule, not by hand.
    agree = t[both_wrong & (t["cnn"] == t["lstm"])]
    examples = []
    for (true, pred), _ in agree.groupby(["true", "cnn"]).size().sort_values(ascending=False).head(4).items():
        rid = agree[(agree["true"] == true) & (agree["cnn"] == pred)].index.min()
        examples.append({"row_id": int(rid), "true": true, "predicted": pred, "prompt": df.loc[rid, "prompt"],
                         "n_tokens": int(t.loc[rid, "n_tokens"]), "snippet": df.loc[rid, "response"][:260]})
    pd.DataFrame(examples).to_csv(OUT / "error_examples.csv", index=False)
    print(stats)
    print(pd.DataFrame(examples)[["true", "predicted", "snippet"]])


if __name__ == "__main__":
    main()
