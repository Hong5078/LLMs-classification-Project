"""Training loop shared by every neural experiment."""
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from .config import read_json, set_seed, write_json
from .evaluate import compute_metrics
from .models import build_model, count_parameters
from .tokenizer import Vocab

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_name():
    return torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu"


def _encode(vocab, token_lists):
    return [torch.tensor(vocab.encode(t) or [vocab.unk_id], dtype=torch.long) for t in token_lists]


def _bucketed_batches(lengths, batch_size, rng, chunk_batches=50):
    """Shuffle, then sort by length inside large chunks so each batch holds sequences of similar length.

    Batch composition and batch order still change every epoch; only the padding per batch is reduced.
    """
    perm = rng.permutation(len(lengths))
    chunk = batch_size * chunk_batches
    batches = []
    for start in range(0, len(perm), chunk):
        part = perm[start:start + chunk]
        part = part[np.argsort(lengths[part], kind="stable")]
        batches += [part[i:i + batch_size] for i in range(0, len(part), batch_size)]
    return [batches[i] for i in rng.permutation(len(batches))]


def _batches(seqs, labels, pad_id, index_batches):
    for idx in index_batches:
        items = [seqs[i] for i in idx]
        lengths = torch.tensor([len(s) for s in items], dtype=torch.long)
        ids = nn.utils.rnn.pad_sequence(items, batch_first=True, padding_value=pad_id)
        yield ids.to(DEVICE), lengths.to(DEVICE), torch.as_tensor(labels[idx], dtype=torch.long).to(DEVICE), idx


@torch.no_grad()
def predict(model, seqs, labels, pad_id, batch_size=256):
    """Return (mean loss, predictions in the original order)."""
    model.eval()
    labels = np.asarray(labels)
    order = np.argsort([len(s) for s in seqs], kind="stable")     # length-sorted batches waste less padding
    index_batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    preds = np.zeros(len(seqs), dtype=int)
    total_loss = 0.0
    for ids, lengths, y, idx in _batches(seqs, labels, pad_id, index_batches):
        logits = model(ids, lengths)
        total_loss += nn.functional.cross_entropy(logits, y, reduction="sum").item()
        preds[idx] = logits.argmax(dim=1).cpu().numpy()
    return total_loss / max(len(seqs), 1), preds


def run_neural(name, out_dir, arch, arch_cfg, cfg, data, class_names, seed, extra_eval=None,
               max_epochs=None, eval_test=True, save_preds=False, force=False, meta=None):
    """Train one model with early stopping on validation macro-F1 and save everything to <out_dir>/<name>.json.

    data: {'train'|'val'|'test': {'tokens', 'labels', 'row_id'}}; extra_eval: {set_name: same structure}.
    The test set (and any extra set) is evaluated exactly once, with the best-validation weights.
    """
    out_path = Path(out_dir) / f"{name}.json"
    if out_path.exists() and not force:
        return read_json(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tc = cfg["train"]
    max_epochs = max_epochs or tc["max_epochs"]
    set_seed(seed)

    vocab = Vocab.build(data["train"]["tokens"], cfg["tokenizer"]["max_vocab"], cfg["tokenizer"]["min_freq"])
    enc = {k: _encode(vocab, v["tokens"]) for k, v in data.items()}
    train_labels = np.asarray(data["train"]["labels"])
    train_lengths = np.array([len(s) for s in enc["train"]])
    n_train = len(train_lengths)

    model = build_model(arch, arch_cfg, len(vocab), len(class_names), vocab.pad_id).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)

    history, best_f1, best_state, best_epoch, bad_epochs = [], -1.0, None, 0, 0
    start = time.time()
    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        model.train()
        index_batches = _bucketed_batches(train_lengths, tc["batch_size"], rng)
        loss_sum, correct = 0.0, 0
        for ids, lengths, y, _ in _batches(enc["train"], train_labels, vocab.pad_id, index_batches):
            optimizer.zero_grad()
            logits = model(ids, lengths)
            loss = criterion(logits, y)
            loss.backward()
            if arch == "lstm":
                nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()

        val_loss, val_pred = predict(model, enc["val"], data["val"]["labels"], vocab.pad_id)
        val_y = np.asarray(data["val"]["labels"])
        val_f1 = f1_score(val_y, val_pred, average="macro")
        history.append({
            "epoch": epoch,
            "train_loss": loss_sum / n_train,
            "train_acc": correct / n_train,
            "val_loss": val_loss,
            "val_acc": float((val_pred == val_y).mean()),
            "val_macro_f1": float(val_f1),
            "seconds": time.time() - epoch_start,
        })
        if val_f1 > best_f1:
            best_f1, best_epoch, bad_epochs = val_f1, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= tc["patience"]:
                break
    train_time = time.time() - start

    model.load_state_dict(best_state)
    result = {
        "name": name, "arch": arch, "arch_cfg": arch_cfg, "train_cfg": tc, "seed": seed,
        "device": device_name(), "vocab_size": len(vocab), "n_types_seen": vocab.n_types_seen,
        "n_params": count_parameters(model),
        "n_train": len(enc["train"]), "n_val": len(enc["val"]),
        "max_seq_len": int(max(len(s) for s in enc["train"])),
        "epochs_run": len(history), "best_epoch": best_epoch,
        "train_time_sec": train_time, "sec_per_epoch": train_time / len(history),
        "history": history, "classes": class_names,
    }
    if meta:
        result.update(meta)

    _, val_pred = predict(model, enc["val"], data["val"]["labels"], vocab.pad_id)
    result["val"] = compute_metrics(data["val"]["labels"], val_pred, class_names)
    eval_sets = dict(extra_eval or {})
    if eval_test:
        eval_sets = {"test": data["test"], **eval_sets}
    for set_name, d in eval_sets.items():
        _, pred = predict(model, _encode(vocab, d["tokens"]), d["labels"], vocab.pad_id)
        result[set_name] = compute_metrics(d["labels"], pred, class_names)
        if save_preds:
            pd.DataFrame({"row_id": d["row_id"], "true": [class_names[i] for i in d["labels"]],
                          "pred": [class_names[i] for i in pred]}).to_csv(
                Path(out_dir) / f"{name}_{set_name}_preds.csv", index=False)

    write_json(result, out_path)
    shown = result["test"] if eval_test else result["val"]
    print(f"[{name}] epochs={len(history)} best={best_epoch} val_f1={best_f1:.4f} "
          f"{'test' if eval_test else 'val'}_acc={shown['accuracy']:.4f} time={train_time:.0f}s", flush=True)
    return result
