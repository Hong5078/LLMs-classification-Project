"""RQ4: stylometric features, a features-only classifier, and the most distinctive n-grams per model.

Usage: python -m src.exp_rq4_features
"""
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import RESULTS, ROOT, load_config, write_json
from .data import load_subset
from .evaluate import compute_metrics
from .features import FEATURES, GROUPS, closing_sentence_start, extract_features, opening_phrase
from .tokenizer import tokenize

OUT = RESULTS / "rq4"


def feature_classifiers(feats, df, cfg):
    models = cfg["models"]
    parts = {s: (feats.loc[df["split"] == s, FEATURES].to_numpy(), df.loc[df["split"] == s, "label"].to_numpy())
             for s in ("train", "val", "test")}
    (Xtr, ytr), (Xva, yva), (Xte, yte) = parts["train"], parts["val"], parts["test"]

    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=cfg["seed"])).fit(Xtr, ytr)
    gb = HistGradientBoostingClassifier(random_state=cfg["seed"]).fit(Xtr, ytr)
    results = {
        "logistic_regression": {"val": compute_metrics(yva, lr.predict(Xva), models),
                                "test": compute_metrics(yte, lr.predict(Xte), models)},
        "gradient_boosting": {"val": compute_metrics(yva, gb.predict(Xva), models),
                              "test": compute_metrics(yte, gb.predict(Xte), models)},
        "n_features": len(FEATURES),
    }

    # Permutation importance on the validation split (drop in accuracy when one feature is shuffled).
    perm = permutation_importance(gb, Xva, yva, n_repeats=10, random_state=cfg["seed"], scoring="accuracy")
    group_of = {f: g for g, fs in GROUPS.items() for f in fs}
    importance = pd.DataFrame({
        "feature": FEATURES, "group": [group_of[f] for f in FEATURES],
        "permutation_importance": perm.importances_mean, "permutation_std": perm.importances_std,
        "lr_mean_abs_coef": np.abs(lr[-1].coef_).mean(axis=0),
    }).sort_values("permutation_importance", ascending=False)
    importance.to_csv(OUT / "feature_importance.csv", index=False)

    # Which feature group carries the signal: train on one group only, and on everything except it.
    rows = []
    for group, names in GROUPS.items():
        only = [FEATURES.index(f) for f in names]
        rest = [i for i in range(len(FEATURES)) if i not in only]
        for label, cols in (("only", only), ("without", rest)):
            clf = HistGradientBoostingClassifier(random_state=cfg["seed"]).fit(Xtr[:, cols], ytr)
            m = compute_metrics(yte, clf.predict(Xte[:, cols]), models)
            rows.append({"group": group, "setting": label, "n_features": len(cols),
                         "test_accuracy": m["accuracy"], "test_macro_f1": m["macro_f1"]})
    groups = pd.DataFrame(rows)
    groups.to_csv(OUT / "feature_groups.csv", index=False)
    write_json(results, OUT / "feature_classifier.json")
    return results, importance, groups


def phrase_table(series, models_col, models, top=5):
    rows = []
    for m in models:
        s = series[models_col == m]
        for phrase, count in Counter(s).most_common(top):
            rows.append({"model": m, "phrase": phrase, "share": count / len(s)})
    return pd.DataFrame(rows)


def distinctive_ngrams(df, models, top=12, min_count=20, prior_tokens=100_000):
    """Log-odds ratio with an informative Dirichlet prior (Monroe et al., 2008), each model against the rest.

    Tokens are lowercased; unigrams and bigrams over words and punctuation. Training split only.
    """
    counts = {m: Counter() for m in models}
    for model, text in zip(df["model"], df["response"]):
        toks = [t.lower() for t in tokenize(text)]
        counts[model].update(toks)
        counts[model].update(" ".join(b) for b in zip(toks, toks[1:]))
    total = Counter()
    for c in counts.values():
        total.update(c)
    vocab = [w for w, c in total.items() if c >= min_count]
    n_total = sum(total[w] for w in vocab)
    prior = {w: total[w] * prior_tokens / n_total for w in vocab}

    rows = []
    for m in models:
        n_i = sum(counts[m][w] for w in vocab)
        n_j = n_total - n_i
        scored = []
        for w in vocab:
            y_i, y_j, a = counts[m][w], total[w] - counts[m][w], prior[w]
            delta = (np.log((y_i + a) / (n_i + prior_tokens - y_i - a))
                     - np.log((y_j + a) / (n_j + prior_tokens - y_j - a)))
            z = delta / np.sqrt(1.0 / (y_i + a) + 1.0 / (y_j + a))
            scored.append((z, w, y_i, y_j))
        for z, w, y_i, y_j in sorted(scored, reverse=True)[:top]:
            rows.append({"model": m, "ngram": w, "z_score": z, "count_model": y_i, "count_others": y_j,
                         "per_10k_model": 1e4 * y_i / n_i, "per_10k_others": 1e4 * y_j / n_j})
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    models = cfg["models"]
    df = load_subset(cfg)
    OUT.mkdir(parents=True, exist_ok=True)

    feats = pd.DataFrame([extract_features(t) for t in df["response"]])
    full = pd.concat([df[["row_id", "model", "split"]], feats], axis=1)
    full.to_csv(ROOT / "data" / "processed" / "features.csv", index=False)

    by_model = feats.groupby(df["model"])
    by_model.mean().loc[models].T.to_csv(OUT / "feature_means.csv")
    by_model.median().loc[models].T.to_csv(OUT / "feature_medians.csv")
    by_model.std().loc[models].T.to_csv(OUT / "feature_stds.csv")
    # Share of responses in which a (mostly zero) formatting feature is present at all.
    (feats > 0).groupby(df["model"]).mean().loc[models].T.to_csv(OUT / "feature_presence.csv")

    results, importance, groups = feature_classifiers(feats, df, cfg)

    phrase_table(df["response"].map(opening_phrase), df["model"], models).to_csv(OUT / "opening_phrases.csv", index=False)
    phrase_table(df["response"].map(closing_sentence_start), df["model"], models).to_csv(
        OUT / "closing_phrases.csv", index=False)

    distinctive_ngrams(df[df["split"] == "train"], models).to_csv(OUT / "distinctive_ngrams.csv", index=False)

    print({k: (v["test"]["accuracy"], v["test"]["macro_f1"]) for k, v in results.items() if isinstance(v, dict)})
    print(importance.head(12))
    print(groups)


if __name__ == "__main__":
    main()
