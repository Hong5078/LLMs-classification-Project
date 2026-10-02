"""Turn the saved result files into LaTeX tables and number macros for the report.

Every number that appears in report/report.tex comes from report/generated/, which this script
writes from results/. Nothing is typed by hand.

Usage: python -m src.make_tables
"""
import pandas as pd

from .config import RESULTS, ROOT, load_config, read_json
from .make_figures import ARCH_LABELS, FEATURE_LABELS, VARIANT_LABELS

GEN = ROOT / "report" / "generated"
MACROS = {}

LATEX_CHARS = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
               "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
               "’": r"\textquoteright{}", "‘": r"\textquoteleft{}", "“": r"\textquotedblleft{}",
               "”": r"\textquotedblright{}", "—": r"\textemdash{}", "–": r"\textendash{}",
               "…": r"\ldots{}", "'": r"\textquotesingle{}", '"': r"\textquotedbl{}",
               "<": r"\textless{}", ">": r"\textgreater{}", "|": r"\textbar{}", "\n": " "}


def tex(s):
    return "".join(LATEX_CHARS.get(c, c if ord(c) < 128 else "?") for c in str(s))


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def num(name, value):
    assert name.isalpha(), name
    MACROS[name] = str(value)


def write(name, lines):
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def tabular(name, spec, header, rows):
    lines = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for r in rows:
        lines.append(r if isinstance(r, str) else " & ".join(str(c) for c in r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(name, lines)


# ------------------------------------------------------------------------------------------------
def dataset_tables(cfg):
    models = cfg["models"]
    log = read_json(RESULTS / "audit" / "cleaning_log.json")
    lengths = pd.read_csv(RESULTS / "audit" / "length_stats.csv", index_col=0)
    ls = read_json(RESULTS / "audit" / "length_summary.json")
    overlap = read_json(RESULTS / "audit" / "prompt_overlap.json")

    rows = []
    for m in models:
        rows.append([m, log["rows_before"][m], log["rows_before"][m] - log["rows_after"][m], log["rows_after"][m],
                     log["split_rows"]["train"][m], log["split_rows"]["val"][m], log["split_rows"]["test"][m],
                     f"{lengths.loc[m, 'n_words_mean']:.0f}", f"{lengths.loc[m, 'n_tokens_mean']:.0f}",
                     f"{lengths.loc[m, 'n_tokens_50%']:.0f}", f"{lengths.loc[m, 'n_tokens_99%']:.0f}"])
    total = lambda split: sum(log["split_rows"][split].values())  # noqa: E731
    rows.append(r"\midrule")
    rows.append(["Total", sum(log["rows_before"].values()), sum(log["rows_before"].values()) - log["total_after"],
                 log["total_after"], total("train"), total("val"), total("test"), "", "", "", ""])
    tabular("tab_dataset.tex", "lrrrrrrrrrr",
            ["Model", "Raw", "Removed", "Kept", "Train", "Val", "Test", "Words", "Tokens", "Median", "P99"], rows)

    num("RawRows", f"{log['raw_rows']:,}")
    num("RawClasses", len(log["raw_sources"]))
    num("SubsetRowsBefore", f"{sum(log['rows_before'].values()):,}")
    num("SubsetRows", f"{log['total_after']:,}")
    num("RemovedRows", sum(log["rows_before"].values()) - log["total_after"])
    num("RemovedSheetErrors", sum(log["spreadsheet_error"].values()))
    num("RemovedDupResponses", sum(log["duplicate_responses_same_model"].values()))
    num("RemovedMissingEmpty", sum(log["missing"].values()) + sum(log["empty"].values()))
    num("DupAcrossModels", log["duplicate_responses_across_models"])
    num("TrainRows", f"{total('train'):,}")
    num("ValRows", f"{total('val'):,}")
    num("TestRows", f"{total('test'):,}")
    num("PromptsRaw", f"{overlap['unique_prompts_raw']:,}")
    num("PromptGroups", f"{overlap['unique_prompt_groups']:,}")
    num("ExactStringsAllSix", f"{overlap['exact_strings_shared_by_all_six']:,}")
    enc = pd.read_csv(RESULTS / "audit" / "prompt_encoding_by_model.csv", index_col=0)
    others = enc.drop(index="Qwen")
    num("QwenCurlyPrompts", int(enc.loc["Qwen", "contains_curly_apostrophe"]))
    num("QwenGarbledPrompts", int(enc.loc["Qwen", "contains_garbled_apostrophe"]))
    num("QwenUniquePrompts", int(enc.loc["Qwen", "exact_string_unique_to_this_model"]))
    num("OthersCurlyPromptsMax", int(others["contains_curly_apostrophe"].max()))
    num("OthersGarbledPromptsMin", int(others["contains_garbled_apostrophe"].min()))
    num("OthersGarbledPromptsMax", int(others["contains_garbled_apostrophe"].max()))
    num("OthersUniquePromptsMax", int(others["exact_string_unique_to_this_model"].max()))
    num("GroupsAllSix", f"{overlap['groups_by_number_of_models'].get('6', 0):,}")
    num("GroupsAllSixPct", pct(overlap["share_of_groups_with_all_six"]))
    num("RowsAllSixPct", pct(overlap["share_of_rows_in_groups_with_all_six"]))
    num("TrainGroups", f"{overlap['groups_per_split']['train']:,}")
    num("ValGroups", overlap["groups_per_split"]["val"])
    num("TestGroups", overlap["groups_per_split"]["test"])
    num("GroupsSharedBetweenSplits", overlap["groups_shared_between_splits"])
    num("CoverageMaxLen", pct(ls["coverage_train"][str(read_json(RESULTS / "timing.json")["max_len"])]))
    num("CoverageThreeEightFour", pct(ls["coverage_train"]["384"]))
    num("CoverageTwoFiveSix", pct(ls["coverage_train"]["256"]))
    num("TokenMedianAll", f"{ls['token_percentiles_all']['0.5']:.0f}")
    num("TokenPNinetyNineAll", f"{ls['token_percentiles_all']['0.99']:.0f}")
    num("TokenMaxAll", f"{ls['token_percentiles_all']['1.0']:.0f}")
    num("PromptTokensMean", f"{ls['prompt_tokens']['mean']:.1f}")
    num("PromptTokensMax", ls["prompt_tokens"]["max"])
    num("ShorterThanSixtyFourPct", pct(ls["share_shorter_than"]["64"], 2))
    num("ShorterThanOneTwentyEightPct", pct(ls["share_shorter_than"]["128"], 1))
    by_tokens = sorted(ls["mean_tokens_by_model"].items(), key=lambda kv: kv[1])
    num("ShortestModel", by_tokens[0][0])
    num("ShortestModelTokens", f"{by_tokens[0][1]:.0f}")
    num("LongestModel", by_tokens[-1][0])
    num("LongestModelTokens", f"{by_tokens[-1][1]:.0f}")

    # audit: giveaways and typography
    self_id = pd.read_csv(RESULTS / "audit" / "self_identification.csv", index_col=0)
    typo = pd.read_csv(RESULTS / "audit" / "typography_rates.csv", index_col=0)
    spec = [
        ("Names its own model family", self_id["names_own_family"]),
        ("Names its own developer", self_id["names_own_developer"]),
        ("Names any of the six families", self_id["names_any_family"]),
        ("AI self-reference phrase", self_id["ai_self_reference_phrase"]),
        (r"\midrule", None),
        ("Curly apostrophe or single quote", typo.loc["curly_apostrophe_or_single_quote"]),
        ("Straight apostrophe", typo.loc["straight_apostrophe"]),
        ("Curly double quote", typo.loc["curly_double_quote"]),
        ("Straight double quote", typo.loc["straight_double_quote"]),
        ("Em dash", typo.loc["em_dash"]),
        (r"\midrule", None),
        ("Newline", typo.loc["newline"]),
        ("Markdown bold", typo.loc["markdown_bold"]),
        ("Markdown header", typo.loc["markdown_header"]),
        ("Backtick", typo.loc["backtick"]),
        (r"LaTeX dollar sign", typo.loc["latex_dollar"]),
        ("No terminal punctuation at the end", typo.loc["ends_without_terminal_punctuation"]),
    ]
    rows = [label if series is None else [label] + [pct(series[m]) for m in models] for label, series in spec]
    tabular("tab_audit.tex", "l" + "r" * len(models), ["Share of responses (\\%) containing"] + models, rows)

    num("MaxOwnFamilyPct", pct(self_id["names_own_family"].max(), 2))
    num("MaxOwnDeveloperPct", pct(self_id["names_own_developer"].max(), 2))
    num("MaxAnyFamilyPct", pct(self_id["names_any_family"].max(), 2))
    num("MaxNewlinePct", pct(typo.loc["newline"].max()))
    num("MaxBoldPct", pct(typo.loc["markdown_bold"].max()))
    for m in models:
        num(f"CurlyApos{m}", pct(typo.loc["curly_apostrophe_or_single_quote", m]))
        num(f"EmDash{m}", pct(typo.loc["em_dash", m]))
        num(f"NoTerminal{m}", pct(typo.loc["ends_without_terminal_punctuation", m]))

    qtype = pd.read_csv(RESULTS / "audit" / "qtype_counts.csv", index_col=0)
    prompts = pd.read_csv(RESULTS / "audit" / "qtype_prompts_per_split.csv", index_col=0)
    rows = [[q.capitalize()] + [qtype.loc[q, m] for m in models]
            + [int(prompts.loc[q, s]) for s in ("train", "val", "test")] for q in qtype.index]
    tabular("tab_qtype.tex", "l" + "r" * len(models) + "|rrr",
            ["Question type"] + models + ["Train", "Val", "Test"], rows)


# ------------------------------------------------------------------------------------------------
def training_tables(cfg):
    timing = read_json(RESULTS / "timing.json")
    best = read_json(RESULTS / "rq1" / "best_config.json")
    tune = pd.read_csv(RESULTS / "tuning" / "tuning_summary.csv")

    rows = []
    for _, r in tune[tune["arch"] == "cnn"].iterrows():
        chosen = r["n_filters"] == best["cnn"]["n_filters"] and r["dropout"] == best["cnn"]["dropout"]
        rows.append(["CNN", f"{int(r['n_filters'])} filters per kernel size, dropout {r['dropout']:.1f}",
                     f"{int(r['n_params']):,}", pct(r["val_accuracy"]), pct(r["val_macro_f1"]),
                     int(r["best_epoch"]), f"{r['sec_per_epoch']:.1f}", r"\checkmark" if chosen else ""])
    rows.append(r"\midrule")
    for _, r in tune[tune["arch"] == "lstm"].iterrows():
        chosen = r["hidden"] == best["lstm"]["hidden"] and r["layers"] == best["lstm"]["layers"]
        rows.append(["BiLSTM", f"hidden size {int(r['hidden'])}, {int(r['layers'])} layer(s)",
                     f"{int(r['n_params']):,}", pct(r["val_accuracy"]), pct(r["val_macro_f1"]),
                     int(r["best_epoch"]), f"{r['sec_per_epoch']:.1f}", r"\checkmark" if chosen else ""])
    tabular("tab_tuning.tex", "llrrrrrc",
            ["Model", "Configuration", "Parameters", "Val acc.", "Val macro-F1", "Best epoch", "s/epoch", "Chosen"],
            rows)

    seed = cfg["seed"]
    cnn = read_json(RESULTS / "rq1" / f"cnn_seed{seed}.json")
    lstm = read_json(RESULTS / "rq1" / f"lstm_seed{seed}.json")
    num("Device", tex(timing["device"]))
    num("MaxLen", timing["max_len"])
    num("BudgetHours", f"{timing['budget_hours']:g}")
    num("TimingCnnSec", f"{timing['candidates'][0]['sec_per_epoch']['cnn']:.1f}")
    num("TimingLstmSec", f"{timing['candidates'][0]['sec_per_epoch']['lstm']:.1f}")
    num("WorstCaseMinutes", f"{timing['candidates'][-1]['worst_case_minutes']:.0f}")
    num("VocabSize", f"{cnn['vocab_size']:,}")
    num("TypesSeen", f"{cnn['n_types_seen']:,}")
    num("MaxVocab", f"{cfg['tokenizer']['max_vocab']:,}")
    num("MinFreq", cfg["tokenizer"]["min_freq"])
    num("EmbDim", best["cnn"]["emb_dim"])
    num("CnnFilters", best["cnn"]["n_filters"])
    num("CnnDropout", f"{best['cnn']['dropout']:g}")
    num("CnnFeatureDim", best["cnn"]["n_filters"] * len(best["cnn"]["kernel_sizes"]))
    num("CnnParams", f"{cnn['n_params']:,}")
    num("LstmHidden", best["lstm"]["hidden"])
    num("LstmLayers", best["lstm"]["layers"])
    num("LstmDropout", f"{best['lstm']['dropout']:g}")
    num("LstmOutDim", 2 * best["lstm"]["hidden"])
    num("LstmFeatureDim", 4 * best["lstm"]["hidden"])
    num("LstmParams", f"{lstm['n_params']:,}")
    tc = cfg["train"]
    num("BatchSize", tc["batch_size"])
    num("LearningRate", f"{tc['lr']:g}")
    num("WeightDecay", f"{tc['weight_decay']:g}")
    num("MaxEpochs", tc["max_epochs"])
    num("Patience", tc["patience"])
    num("GradClip", f"{tc['grad_clip']:g}")
    num("InputMaxLen", cfg["lengths"]["input"])
    num("PromptInConcat", cfg["lengths"]["prompt_in_concat"])
    num("CnnBestEpoch", cnn["best_epoch"])
    num("CnnEpochsRun", cnn["epochs_run"])
    num("CnnTrainSeconds", f"{cnn['train_time_sec']:.0f}")
    num("CnnSecPerEpoch", f"{cnn['sec_per_epoch']:.1f}")
    num("LstmBestEpoch", lstm["best_epoch"])
    num("LstmEpochsRun", lstm["epochs_run"])
    num("LstmTrainSeconds", f"{lstm['train_time_sec']:.0f}")
    num("LstmSecPerEpoch", f"{lstm['sec_per_epoch']:.1f}")
    num("CnnFinalTrainAcc", pct(cnn["history"][cnn["best_epoch"] - 1]["train_acc"]))
    num("LstmFinalTrainAcc", pct(lstm["history"][lstm["best_epoch"] - 1]["train_acc"]))
    num("CnnFinalValAcc", pct(cnn["history"][cnn["best_epoch"] - 1]["val_acc"]))
    num("LstmFinalValAcc", pct(lstm["history"][lstm["best_epoch"] - 1]["val_acc"]))

    base = read_json(RESULTS / "rq1" / "baseline_output.json")
    num("BaselineC", f"{base['C']:g}")
    num("BaselineFeatures", f"{base['n_features']:,}")

    # total training time of all saved neural runs
    total = 0.0
    n_runs = 0
    for path in RESULTS.rglob("*.json"):
        if path.parent.name == "timing":
            continue
        try:
            r = read_json(path)
        except Exception:
            continue
        if isinstance(r, dict) and "train_time_sec" in r:
            total += r["train_time_sec"]
            n_runs += 1
    num("TotalTrainMinutes", f"{total / 60:.0f}")
    num("TotalRuns", n_runs)


# ------------------------------------------------------------------------------------------------
def rq1_tables(cfg):
    models = cfg["models"]
    seed = cfg["seed"]
    s = pd.read_csv(RESULTS / "rq1" / "summary.csv", index_col=0)
    base = read_json(RESULTS / "rq1" / "baseline_output.json")["test"]
    chance = 1 / len(models)

    def pm(arch, col):
        return f"{pct(s.loc[arch, col + '_mean'])} $\\pm$ {pct(s.loc[arch, col + '_std'])}"

    rows = []
    for arch in ("cnn", "lstm"):
        rows.append([ARCH_LABELS[arch], pm(arch, "test_accuracy"), pm(arch, "test_macro_f1")]
                    + [pct(s.loc[arch, f"f1_{m}_mean"]) for m in models])
    rows.append(r"\midrule")
    rows.append(["TF-IDF + LR", pct(base["accuracy"]), pct(base["macro_f1"])]
                + [pct(base["per_class_f1"][m]) for m in models])
    rows.append(["Chance", pct(chance), "--"] + ["--"] * len(models))
    tabular("tab_rq1.tex", "lcc" + "r" * len(models), ["Model", "Accuracy", "Macro-F1"] + models, rows)

    for arch, tag in (("cnn", "Cnn"), ("lstm", "Lstm")):
        num(f"RqOne{tag}Acc", pct(s.loc[arch, "test_accuracy_mean"]))
        num(f"RqOne{tag}AccStd", pct(s.loc[arch, "test_accuracy_std"]))
        num(f"RqOne{tag}Fone", pct(s.loc[arch, "test_macro_f1_mean"]))
        num(f"RqOne{tag}FoneStd", pct(s.loc[arch, "test_macro_f1_std"]))
        f1 = {m: s.loc[arch, f"f1_{m}_mean"] for m in models}
        ranked = sorted(f1, key=f1.get)
        num(f"RqOne{tag}Hardest", ranked[0])
        num(f"RqOne{tag}HardestFone", pct(f1[ranked[0]]))
        num(f"RqOne{tag}SecondHardest", ranked[1])
        num(f"RqOne{tag}SecondHardestFone", pct(f1[ranked[1]]))
        num(f"RqOne{tag}Easiest", ranked[-1])
        num(f"RqOne{tag}EasiestFone", pct(f1[ranked[-1]]))
        num(f"RqOne{tag}SecondEasiest", ranked[-2])
        num(f"RqOne{tag}SecondEasiestFone", pct(f1[ranked[-2]]))
        r42 = read_json(RESULTS / "rq1" / f"{arch}_seed{seed}.json")["test"]
        num(f"RqOne{tag}SeedAcc", pct(r42["accuracy"]))
        cm = pd.DataFrame(read_json(RESULTS / "rq1" / f"{arch}_mean_confusion.json")["mean_confusion_matrix_normalized"],
                          index=models, columns=models)
        off = cm.where(~pd.DataFrame(cm.index.values[:, None] == cm.columns.values[None, :],
                                     index=models, columns=models)).stack().sort_values(ascending=False)
        for k, word in enumerate(("One", "Two", "Three")):
            (true, pred), v = off.index[k], off.iloc[k]
            num(f"RqOne{tag}Conf{word}True", true)
            num(f"RqOne{tag}Conf{word}Pred", pred)
            num(f"RqOne{tag}Conf{word}Pct", pct(v))
    num("BaselineAcc", pct(base["accuracy"]))
    num("BaselineFone", pct(base["macro_f1"]))
    num("ChancePct", pct(chance))
    num("NumSeeds", len(cfg["seeds_rq1"]))

    # error analysis
    es = read_json(RESULTS / "rq1" / "error_stats.json")
    num("ErrCnn", es["cnn_errors"])
    num("ErrLstm", es["lstm_errors"])
    num("ErrBoth", es["both_wrong"])
    num("ErrBothSame", es["both_wrong_same_prediction"])
    num("ErrEither", es["either_wrong"])
    num("ErrBothShareOfCnn", pct(es["both_wrong"] / es["cnn_errors"], 0))
    num("ErrTokensCnnCorrect", f"{es['mean_tokens_cnn_correct']:.0f}")
    num("ErrTokensCnnWrong", f"{es['mean_tokens_cnn_wrong']:.0f}")
    num("ErrTokensLstmCorrect", f"{es['mean_tokens_lstm_correct']:.0f}")
    num("ErrTokensLstmWrong", f"{es['mean_tokens_lstm_wrong']:.0f}")
    ex = pd.read_csv(RESULTS / "rq1" / "error_examples.csv")
    rows = [[r["true"], r["predicted"], r"\textit{" + tex(r["prompt"]) + r"} " + tex(r["snippet"]) + r"\ldots"]
            for _, r in ex.iterrows()]
    tabular("tab_errors.tex", r"llp{0.68\linewidth}", ["True", "Predicted", "Prompt (italic) and start of the response"],
            rows)
    pairs = pd.read_csv(RESULTS / "rq1" / "error_pairs.csv").head(6)
    tabular("tab_error_pairs.tex", "llrr", ["True model", "Predicted as", "CNN errors", "BiLSTM errors"],
            [[r["true"], r["predicted"], r["cnn_errors"], r["lstm_errors"]] for _, r in pairs.iterrows()])


# ------------------------------------------------------------------------------------------------
def rq2_tables(cfg):
    s = pd.read_csv(RESULTS / "rq2" / "summary.csv")
    cond = {"input": "Input only", "input_ascii": "Input only, ASCII prompt", "output": "Output only",
            "input+output": "Input + output"}
    chance = pct(1 / len(cfg["models"]))
    rows = []
    for arch in ("cnn", "lstm", "tfidf_lr"):
        row = [ARCH_LABELS[arch]]
        for c in cond:
            r = s[(s["arch"] == arch) & (s["condition"] == c)].iloc[0]
            row += [pct(r["test_accuracy"]), pct(r["test_macro_f1"])]
            tag = {"cnn": "Cnn", "lstm": "Lstm", "tfidf_lr": "Base"}[arch]
            ctag = {"input": "In", "input_ascii": "InAscii", "output": "Out", "input+output": "Both"}[c]
            num(f"RqTwo{tag}{ctag}Acc", pct(r["test_accuracy"]))
            num(f"RqTwo{tag}{ctag}Fone", pct(r["test_macro_f1"]))
        rows.append(row)
    rows.append(r"\midrule")
    rows.append(["Chance"] + [chance, "--"] * len(cond))
    lines = [r"\begin{tabular}{l" + "cc" * len(cond) + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{label}}}" for label in cond.values()) + r" \\",
             "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(cond))),
             "Model & " + " & ".join(["Acc.", "F1"] * len(cond)) + r" \\", r"\midrule"]
    lines += [r if isinstance(r, str) else " & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_rq2.tex", lines)

    b = read_json(RESULTS / "rq2" / "input_only_breakdown.json")
    num("RqTwoSharedN", f"{b['cnn']['n_shared']:,}")
    num("RqTwoNotSharedN", b["cnn"]["n_not_shared"])
    num("RqTwoNotSharedQwen", b["cnn"]["not_shared_by_true_model"].get("Qwen", 0))
    for arch, tag in (("cnn", "Cnn"), ("lstm", "Lstm"), ("tfidf_lr", "Base")):
        num(f"RqTwo{tag}SharedAcc", pct(b[arch]["accuracy_shared"]))
        num(f"RqTwo{tag}NotSharedAcc", pct(b[arch]["accuracy_not_shared"]))
        num(f"RqTwo{tag}NotSharedQwenCorrect", b[arch]["not_shared_correct_by_true_model"].get("Qwen", 0))
    for arch, tag in (("cnn", "Cnn"), ("lstm", "Lstm"), ("tfidf_lr", "Base")):
        get = lambda c: s[(s["arch"] == arch) & (s["condition"] == c)]["test_accuracy"].iloc[0]  # noqa: E731
        num(f"RqTwo{tag}BothMinusOut", f"{100 * (get('input+output') - get('output')):+.1f}")


# ------------------------------------------------------------------------------------------------
def rq3_tables(cfg):
    models = cfg["models"]
    t = pd.read_csv(RESULTS / "rq3" / "transfer.csv")
    runs = pd.read_csv(RESULTS / "rq3" / "runs.csv")
    rows = []
    for arch in ("cnn", "lstm", "tfidf_lr"):
        for domain in ("definitional", "explanatory"):
            r = t[(t["arch"] == arch) & (t["test_domain"] == domain)].iloc[0]
            rows.append([ARCH_LABELS[arch], domain.capitalize(), int(r["n_test"]), pct(r["in_accuracy"]),
                         pct(r["cross_accuracy"]), f"{100 * r['accuracy_drop']:+.1f}", pct(r["in_macro_f1"]),
                         pct(r["cross_macro_f1"]), f"{100 * r['macro_f1_drop']:+.1f}"])
            tag = {"cnn": "Cnn", "lstm": "Lstm", "tfidf_lr": "Base"}[arch] + domain[:3].capitalize()
            num(f"RqThree{tag}In", pct(r["in_accuracy"]))
            num(f"RqThree{tag}Cross", pct(r["cross_accuracy"]))
            num(f"RqThree{tag}Drop", f"{100 * r['accuracy_drop']:.1f}")
        if arch != "tfidf_lr":
            rows.append(r"\midrule")
    tabular("tab_rq3.tex", "llrrrrrrr",
            ["Model", "Test question type", "$n$", "In acc.", "Cross acc.", "Drop", "In F1", "Cross F1", "Drop"], rows)

    rows = []
    for arch in ("cnn", "lstm"):
        for domain in ("definitional", "explanatory"):
            r = t[(t["arch"] == arch) & (t["test_domain"] == domain)].iloc[0]
            rows.append([ARCH_LABELS[arch], domain.capitalize()] + [f"{100 * r[f'f1_drop_{m}']:+.1f}" for m in models])
    tabular("tab_rq3_perclass.tex", "ll" + "r" * len(models), ["Model", "Test question type"] + models, rows)

    mean_drop = {m: t[t["arch"].isin(["cnn", "lstm"])][f"f1_drop_{m}"].mean() for m in models}
    ranked = sorted(mean_drop, key=mean_drop.get)
    num("RqThreeMostStable", ranked[0])
    num("RqThreeMostStableDrop", f"{100 * mean_drop[ranked[0]]:.1f}")
    num("RqThreeLeastStable", ranked[-1])
    num("RqThreeLeastStableDrop", f"{100 * mean_drop[ranked[-1]]:.1f}")
    num("RqThreeSecondLeastStable", ranked[-2])
    num("RqThreeSecondLeastStableDrop", f"{100 * mean_drop[ranked[-2]]:.1f}")
    nn = t[t["arch"].isin(["cnn", "lstm"])]
    num("RqThreeMeanDropNeural", f"{100 * nn['accuracy_drop'].mean():.1f}")
    num("RqThreeMaxDropNeural", f"{100 * nn['accuracy_drop'].max():.1f}")
    num("RqThreeMinDropNeural", f"{100 * nn['accuracy_drop'].min():.1f}")
    m_rows = runs[runs["train_set"] == "explanatory"].iloc[0]
    num("RqThreeMatchedTrainRows", f"{int(m_rows['n_train']):,}")
    full = runs[runs["train_set"] == "definitional_full"]
    num("RqThreeFullTrainRows", f"{int(full.iloc[0]['n_train']):,}")
    for arch, tag in (("cnn", "Cnn"), ("lstm", "Lstm"), ("tfidf_lr", "Base")):
        r = full[full["arch"] == arch].iloc[0]
        num(f"RqThree{tag}FullIn", pct(r["in_accuracy"]))
        num(f"RqThree{tag}FullCross", pct(r["cross_accuracy"]))
    qtype = pd.read_csv(RESULTS / "audit" / "qtype_counts.csv", index_col=0)
    for q, tag in (("definitional", "Def"), ("explanatory", "Exp"), ("mixed", "Mixed"), ("other", "Other")):
        num(f"Qtype{tag}Rows", f"{int(qtype.loc[q].sum()):,}")


# ------------------------------------------------------------------------------------------------
def rq4_tables(cfg):
    models = cfg["models"]
    fc = read_json(RESULTS / "rq4" / "feature_classifier.json")
    num("FeatCount", fc["n_features"])
    num("FeatLrAcc", pct(fc["logistic_regression"]["test"]["accuracy"]))
    num("FeatLrFone", pct(fc["logistic_regression"]["test"]["macro_f1"]))
    num("FeatGbAcc", pct(fc["gradient_boosting"]["test"]["accuracy"]))
    num("FeatGbFone", pct(fc["gradient_boosting"]["test"]["macro_f1"]))

    g = pd.read_csv(RESULTS / "rq4" / "feature_groups.csv")
    names = {"length": "Length", "lexical": "Lexical", "punctuation": "Punctuation",
             "typography": "Typography (quote and dash style)", "formatting": "Formatting (markdown, newlines)"}
    rows = []
    for group, label in names.items():
        only = g[(g["group"] == group) & (g["setting"] == "only")].iloc[0]
        without = g[(g["group"] == group) & (g["setting"] == "without")].iloc[0]
        rows.append([label, int(only["n_features"]), pct(only["test_accuracy"]), pct(without["test_accuracy"]),
                     f"{100 * (fc['gradient_boosting']['test']['accuracy'] - without['test_accuracy']):+.1f}"])
        num(f"FeatOnly{group.capitalize()}", pct(only["test_accuracy"]))
        num(f"FeatWithout{group.capitalize()}", pct(without["test_accuracy"]))
    rows.append(r"\midrule")
    rows.append(["All features", fc["n_features"], pct(fc["gradient_boosting"]["test"]["accuracy"]), "--", "--"])
    tabular("tab_feature_groups.tex", "lrrrr",
            ["Feature group", "Features", "Only this group", "All except it", "Loss when removed"], rows)

    imp = pd.read_csv(RESULTS / "rq4" / "feature_importance.csv")
    for k, word in enumerate(("One", "Two", "Three", "Four", "Five")):
        num(f"FeatTop{word}", FEATURE_LABELS[imp.iloc[k]["feature"]].lower())
        num(f"FeatTop{word}Imp", pct(imp.iloc[k]["permutation_importance"]))

    means = pd.read_csv(RESULTS / "rq4" / "feature_means.csv", index_col=0)
    for m in models:
        num(f"Words{m}", f"{means.loc['n_words', m]:.0f}")
        num(f"WordsPerSentence{m}", f"{means.loc['words_per_sentence', m]:.1f}")
        num(f"Mattr{m}", f"{means.loc['mattr50', m]:.3f}")
        num(f"Ttr{m}", f"{means.loc['ttr', m]:.3f}")
        num(f"ParenRate{m}", f"{means.loc['paren_rate', m]:.2f}")
        num(f"AvgWordLen{m}", f"{means.loc['avg_word_len', m]:.2f}")
        num(f"EmDashRate{m}", f"{means.loc['em_dash_rate', m]:.2f}")

    ng = pd.read_csv(RESULTS / "rq4" / "distinctive_ngrams.csv", keep_default_na=False)
    rows = []
    for m in models:
        top = ng[ng["model"] == m].head(8)
        rows.append([m, ", ".join(r"\texttt{" + tex(w) + "}" for w in top["ngram"])])
    tabular("tab_ngrams.tex", r"lp{0.82\linewidth}", ["Model", "Most over-represented n-grams (highest z-score first)"],
            rows)

    closing = pd.read_csv(RESULTS / "rq4" / "closing_phrases.csv", keep_default_na=False)
    opening = pd.read_csv(RESULTS / "rq4" / "opening_phrases.csv", keep_default_na=False)
    rows = []
    for m in models:
        c = closing[closing["model"] == m].head(3)
        o = opening[opening["model"] == m].head(2)
        rows.append([m,
                     "; ".join(f"{tex(p)} ({pct(s)}\\%)" for p, s in zip(o["phrase"], o["share"])),
                     "; ".join(f"{tex(p)} ({pct(s)}\\%)" for p, s in zip(c["phrase"], c["share"]))])
    tabular("tab_phrases.tex", r"lp{0.36\linewidth}p{0.46\linewidth}",
            ["Model", "Most common first three words", "Most common first two words of the last sentence"], rows)

    # ablations
    a = pd.read_csv(RESULTS / "rq4" / "ablations.csv")
    archs = [x for x in ("cnn", "lstm", "tfidf_lr") if (a["arch"] == x).sum() > 1]
    rows = []
    for v, label in VARIANT_LABELS.items():
        if v not in set(a["variant"]):
            continue
        sub = a[a["variant"] == v].set_index("arch")
        row = [label, "--" if v == "none" else pct(sub["share_changed"].iloc[0])]
        for arch in archs:
            if arch not in sub.index:
                row += ["--", "--"]
                continue
            row += [pct(sub.loc[arch, "test_accuracy"]),
                    "--" if v == "none" else f"{-100 * sub.loc[arch, 'accuracy_drop']:+.1f}"]
            if v != "none":
                tag = {"cnn": "Cnn", "lstm": "Lstm", "tfidf_lr": "Base"}[arch]
                vtag = "".join(w.capitalize() for w in v.replace("64", "SixtyFour").replace("128", "OneTwentyEight")
                               .split("_"))
                num(f"Abl{tag}{vtag}Acc", pct(sub.loc[arch, "test_accuracy"]))
                num(f"Abl{tag}{vtag}Drop", f"{100 * sub.loc[arch, 'accuracy_drop']:.1f}")
        if v != "none":
            vtag = "".join(w.capitalize() for w in v.replace("64", "SixtyFour").replace("128", "OneTwentyEight")
                           .split("_"))
            num(f"Abl{vtag}Changed", pct(sub["share_changed"].iloc[0]))
        rows.append(row)
        if v == "none":
            rows.append(r"\midrule")
    lines = [r"\begin{tabular}{lr" + "rr" * len(archs) + "}", r"\toprule",
             " & Responses & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{ARCH_LABELS[x]}}}" for x in archs) + r" \\",
             "".join(rf"\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(archs))),
             r"Responses modified & changed (\%) & " + " & ".join(["Acc.", r"$\Delta$"] * len(archs)) + r" \\",
             r"\midrule"]
    lines += [r if isinstance(r, str) else " & ".join(str(c) for c in r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_ablations.tex", lines)


def main():
    cfg = load_config()
    for fn in (dataset_tables, training_tables, rq1_tables, rq2_tables, rq3_tables, rq4_tables):
        fn(cfg)
    write("numbers.tex", [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in sorted(MACROS.items())])
    print(f"wrote {len(MACROS)} number macros and tables to {GEN}")


if __name__ == "__main__":
    main()
