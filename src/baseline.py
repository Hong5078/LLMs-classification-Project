"""TF-IDF (word + character n-grams) + logistic regression reference model."""
import time
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion

from .config import RESULTS, read_json, write_json
from .evaluate import compute_metrics


def _vectorizer(bc):
    common = dict(lowercase=False, sublinear_tf=True, min_df=2, max_features=bc["max_features"])
    return FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", token_pattern=r"\n|[^\W\d_]+|\d+|[^\w\s]",
                                 ngram_range=tuple(bc["word_ngram"]), **common)),
        ("char", TfidfVectorizer(analyzer="char", ngram_range=tuple(bc["char_ngram"]), **common)),
    ])


def chosen_C(cfg):
    """C selected on validation in RQ1; every other baseline run reuses it."""
    path = RESULTS / "rq1" / "baseline_output.json"
    return read_json(path)["C"] if path.exists() else cfg["baseline"]["C_grid"][0]


def run_baseline(name, out_dir, cfg, data, class_names, extra_eval=None, tune_C=False, force=False, meta=None,
                 save_preds=False):
    out_path = Path(out_dir) / f"{name}.json"
    if out_path.exists() and not force:
        return read_json(out_path)

    bc = cfg["baseline"]
    start = time.time()
    vec = _vectorizer(bc)
    X_train = vec.fit_transform(data["train"]["strings"])
    X_val = vec.transform(data["val"]["strings"])
    y_train, y_val = data["train"]["labels"], data["val"]["labels"]

    grid = bc["C_grid"] if tune_C else [chosen_C(cfg)]
    best, tuning = None, []
    for C in grid:
        clf = LogisticRegression(C=C, max_iter=1000, random_state=cfg["seed"]).fit(X_train, y_train)
        f1 = f1_score(y_val, clf.predict(X_val), average="macro")
        tuning.append({"C": C, "val_macro_f1": float(f1)})
        if best is None or f1 > best[0]:
            best = (f1, C, clf)
    _, C, clf = best

    result = {"name": name, "arch": "tfidf_lr", "C": C, "tuning": tuning, "n_features": X_train.shape[1],
              "n_train": X_train.shape[0], "classes": class_names,
              "val": compute_metrics(y_val, clf.predict(X_val), class_names)}
    if meta:
        result.update(meta)
    for set_name, d in {"test": data["test"], **(extra_eval or {})}.items():
        pred = clf.predict(vec.transform(d["strings"]))
        result[set_name] = compute_metrics(d["labels"], pred, class_names)
        if save_preds:
            pd.DataFrame({"row_id": d["row_id"], "true": [class_names[i] for i in d["labels"]],
                          "pred": [class_names[i] for i in pred]}).to_csv(
                Path(out_dir) / f"{name}_{set_name}_preds.csv", index=False)
    result["train_time_sec"] = time.time() - start
    write_json(result, out_path)
    print(f"[{name}] C={C} val_f1={result['val']['macro_f1']:.4f} test_acc={result['test']['accuracy']:.4f} "
          f"time={result['train_time_sec']:.0f}s", flush=True)
    return result
