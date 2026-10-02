"""Time one epoch of each model and pick the response max length that fits the compute budget.

Usage: python -m src.exp_timing
"""
from .config import RESULTS, load_config, write_json
from .data import build_inputs, by_split, load_subset
from .train import device_name, run_neural

# Full-length training runs that the required parts of the pipeline need, in units of one
# default-size run. CNN: 4 tuning + 3 seeds + 1 input+output + 5 full-length ablations + 1 for all
# short runs (input-only, truncated ablations, RQ3 subsets). LSTM: 5 tuning (two of the four grid
# points have two layers) + 3 seeds + 1 input+output + 1 for short runs. The LSTM ablations are
# optional ("if time allows") and are not part of this worst-case budget.
FULL_RUNS = {"cnn": 14, "lstm": 10}
CANDIDATE_LENGTHS = (512, 384, 256)


def main():
    cfg = load_config()
    df = load_subset(cfg)
    budget_sec = cfg["budget_hours"] * 3600
    report = {"device": device_name(), "budget_hours": cfg["budget_hours"], "candidates": []}

    chosen = None
    for max_len in CANDIDATE_LENGTHS:
        if max_len > cfg["lengths"]["output"]:
            continue
        data = by_split(df, *build_inputs(df, cfg, "output", out_len=max_len))
        secs = {}
        for arch in ("cnn", "lstm"):
            r = run_neural(f"timing_{arch}_len{max_len}", RESULTS / "timing", arch, cfg[arch], cfg, data,
                           cfg["models"], cfg["seed"], max_epochs=1, eval_test=False)
            secs[arch] = r["sec_per_epoch"]
        # Worst case: every run uses all max_epochs (early stopping normally ends runs much sooner).
        worst_case = cfg["train"]["max_epochs"] * sum(FULL_RUNS[a] * secs[a] for a in FULL_RUNS)
        report["candidates"].append({"max_len": max_len, "sec_per_epoch": secs,
                                     "worst_case_minutes": worst_case / 60})
        if worst_case <= budget_sec:
            chosen = max_len
            break
    report["max_len"] = chosen or CANDIDATE_LENGTHS[-1]
    report["full_runs"] = FULL_RUNS
    write_json(report, RESULTS / "timing.json")
    print(report)


if __name__ == "__main__":
    main()
