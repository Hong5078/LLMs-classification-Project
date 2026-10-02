"""Draw every figure in the report from the saved result files.

Usage: python -m src.make_figures
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .config import FIGURES, RESULTS, load_config, read_json  # noqa: E402

# Colour follows the entity: each LLM keeps one colour in every figure. The order was checked with a
# colour-vision-deficiency validator (adjacent pairs); model names are always written next to the marks.
MODEL_COLORS = {"ChatGPT": "#2a78d6", "Gemini": "#eb6834", "Claude": "#1baf7a",
                "Llama": "#eda100", "DeepSeek": "#e87ba4", "Qwen": "#008300"}
ARCH_COLORS = {"cnn": "#2a78d6", "lstm": "#eb6834", "tfidf_lr": "#1baf7a"}
ARCH_LABELS = {"cnn": "CNN", "lstm": "BiLSTM", "tfidf_lr": "TF-IDF + LR"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#ffffff", "#cde2fb", "#6da7ec", "#2a78d6", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])

VARIANT_LABELS = {
    "none": "Original text",
    "strip_markdown": "(a) Markdown stripped",
    "normalize_whitespace": "(b) Whitespace normalised",
    "lower_nopunct": "(c) Lowercase, no punctuation",
    "truncate_128": "(d) First 128 tokens",
    "truncate_64": "(d) First 64 tokens",
    "mask_self_id": "(e) Model names masked",
    "normalize_typography": "(f) Typography normalised",
    "lower_nopunct_truncate_64": "(c)+(d) Words only, first 64",
}


def style():
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlesize": 10,
        "axes.titleweight": "semibold", "axes.titlecolor": INK, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.major.size": 0, "ytick.major.size": 0, "legend.frameon": False, "legend.fontsize": 8.5,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.bbox": "tight", "savefig.dpi": 200,
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "pdf.fonttype": 42,
    })


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png")
    plt.close(fig)


def fig_lengths(cfg):
    df = pd.read_csv(RESULTS / "audit" / "lengths_per_row.csv")
    max_len = read_json(RESULTS / "timing.json")["max_len"]
    models = cfg["models"][::-1]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    data = [df.loc[df["model"] == m, "n_tokens"] for m in models]
    bp = ax.boxplot(data, vert=False, widths=0.5, patch_artist=True, whis=(1, 99), showfliers=True,
                    flierprops=dict(marker=".", markersize=2, markeredgecolor=MUTED, alpha=0.5),
                    medianprops=dict(color="white", linewidth=1.5), whiskerprops=dict(color=AXIS),
                    capprops=dict(color=AXIS))
    for patch, m in zip(bp["boxes"], models):
        patch.set(facecolor=MODEL_COLORS[m], edgecolor="none")
    ax.axvline(max_len, color=INK2, linewidth=1)
    ax.text(max_len + 8, len(models) + 0.35, f"max length = {max_len}", color=INK2, fontsize=8, va="top")
    ax.set_yticks(range(1, len(models) + 1), models)
    ax.set_xlabel("Response length (tokens)")
    ax.set_xlim(0, min(df["n_tokens"].max() * 1.02, 1000))
    ax.grid(axis="y", visible=False)
    save(fig, "length_distribution")


def fig_training_curves(cfg):
    seed = cfg["seed"]
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.2), sharex="col")
    for col, arch in enumerate(("cnn", "lstm")):
        r = read_json(RESULTS / "rq1" / f"{arch}_seed{seed}.json")
        h = pd.DataFrame(r["history"])
        for row, (a, b, ylabel) in enumerate((("train_loss", "val_loss", "Cross-entropy loss"),
                                             ("train_acc", "val_acc", "Accuracy"))):
            ax = axes[row, col]
            ax.plot(h["epoch"], h[a], color="#2a78d6", label="Train", marker="o", markersize=3.5)
            ax.plot(h["epoch"], h[b], color="#eb6834", label="Validation", marker="o", markersize=3.5)
            ax.axvline(r["best_epoch"], color=AXIS, linewidth=1)
            if row == 0:
                ax.set_title(ARCH_LABELS[arch])
                ax.text(r["best_epoch"], ax.get_ylim()[1], " best epoch", color=MUTED, fontsize=7.5, va="top")
            else:
                ax.set_xlabel("Epoch")
            if col == 0:
                ax.set_ylabel(ylabel)
            ax.set_xticks(h["epoch"])
    axes[0, 0].legend(loc="upper right")
    fig.tight_layout()
    save(fig, "training_curves")


def _confusion(ax, cm, labels, title):
    cm = np.asarray(cm)
    ax.imshow(cm, cmap=SEQ, vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if cm[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title(title)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def fig_confusion(cfg):
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.3), sharey=True)
    for ax, arch in zip(axes, ("cnn", "lstm")):
        r = read_json(RESULTS / "rq1" / f"{arch}_mean_confusion.json")
        _confusion(ax, r["mean_confusion_matrix_normalized"], r["classes"], ARCH_LABELS[arch])
        ax.set_xlabel("Predicted model")
    axes[0].set_ylabel("True model")
    fig.tight_layout()
    save(fig, "confusion_rq1")


def fig_rq2(cfg):
    s = pd.read_csv(RESULTS / "rq2" / "summary.csv")
    conditions = ["input", "output", "input+output"]
    labels = ["Input only", "Output only", "Input + output"]
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    width = 0.22
    for k, arch in enumerate(("cnn", "lstm", "tfidf_lr")):
        vals = [s[(s["arch"] == arch) & (s["condition"] == c)]["test_accuracy"].iloc[0] for c in conditions]
        x = np.arange(len(conditions)) + (k - 1) * (width + 0.02)
        ax.bar(x, vals, width, color=ARCH_COLORS[arch], label=ARCH_LABELS[arch])
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    chance = 1 / len(cfg["models"])
    ax.axhline(chance, color=INK2, linewidth=1)
    ax.text(len(conditions) - 0.5, chance + 0.012, "chance (1/6)", color=INK2, fontsize=7.5, ha="right")
    ax.set_xticks(range(len(conditions)), labels)
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=3)
    save(fig, "rq2_conditions")


def fig_rq3(cfg):
    t = pd.read_csv(RESULTS / "rq3" / "transfer.csv")
    models = cfg["models"][::-1]
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.4), sharex=True, sharey=True)
    for row, arch in enumerate(("cnn", "lstm")):
        for col, domain in enumerate(("definitional", "explanatory")):
            ax = axes[row, col]
            r = t[(t["arch"] == arch) & (t["test_domain"] == domain)].iloc[0]
            y = np.arange(len(models))
            ind = [r[f"in_f1_{m}"] for m in models]
            cross = [r[f"cross_f1_{m}"] for m in models]
            ax.hlines(y, cross, ind, color=AXIS, linewidth=1.5, zorder=1)
            ax.scatter(ind, y, s=34, color="#2a78d6", edgecolor="white", linewidth=1, zorder=3,
                       label="Trained on the same question type")
            ax.scatter(cross, y, s=34, color="#eb6834", edgecolor="white", linewidth=1, zorder=3,
                       label="Trained on the other question type")
            ax.set_yticks(y, models)
            ax.set_title(f"{ARCH_LABELS[arch]}, tested on {domain}", fontsize=9)
            ax.grid(axis="y", visible=False)
            if row == 1:
                ax.set_xlabel("Per-class F1 on the test split")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    save(fig, "rq3_transfer")


FEATURE_LABELS = {
    "n_words": "Words", "words_per_sentence": "Words per sentence", "n_sentences": "Sentences",
    "avg_word_len": "Mean word length", "ttr": "Type-token ratio", "mattr50": "MATTR (window 50)",
    "comma_rate": "Commas / 100 words", "semicolon_rate": "Semicolons / 100 words",
    "colon_rate": "Colons / 100 words", "paren_rate": "Parentheses / 100 words",
    "hyphen_rate": "Hyphens / 100 words", "curly_apos_rate": "Curly apostrophes / 100 words",
    "straight_apos_rate": "Straight apostrophes / 100 words", "em_dash_rate": "Em dashes / 100 words",
    "straight_dquote_rate": "Straight double quotes / 100 words", "period_rate": "Periods / 100 words",
    "n_chars": "Characters", "non_ascii_rate": "Non-ASCII character share",
    "curly_dquote_rate": "Curly double quotes / 100 words", "en_dash_rate": "En dashes / 100 words",
    "pron_2p_rate": "Second-person pronouns / 100 words", "pron_1p_rate": "First-person pronouns / 100 words",
    "upper_rate": "Uppercase letter share", "digit_rate": "Digit share", "latex_rate": "LaTeX markers / 100 words",
    "inline_numbered": "Inline numbered items", "ends_terminal": "Ends with . ! or ?",
    "question_rate": "Question marks / 100 words", "exclaim_rate": "Exclamation marks / 100 words",
    "slash_rate": "Slashes / 100 words", "md_bold": "Bold spans", "md_code": "Backticks",
    "md_headers": "Markdown headers", "md_bullets": "Bullet lines", "md_numbered": "Numbered-list lines",
    "md_table_rows": "Table rows", "emoji_count": "Emoji", "newline_rate": "Newlines / 100 words",
}
PROFILE_FEATURES = ["n_words", "n_sentences", "words_per_sentence", "avg_word_len", "ttr", "mattr50",
                    "comma_rate", "semicolon_rate", "colon_rate", "paren_rate", "hyphen_rate",
                    "curly_apos_rate", "straight_apos_rate", "em_dash_rate", "straight_dquote_rate"]


def fig_feature_profiles(cfg):
    means = pd.read_csv(RESULTS / "rq4" / "feature_means.csv", index_col=0)
    models = cfg["models"]
    m = means.loc[PROFILE_FEATURES, models]
    z = m.sub(m.mean(axis=1), axis=0).div(m.std(axis=1, ddof=0), axis=0)
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    im = ax.imshow(z.to_numpy(), cmap=DIV, vmin=-2.3, vmax=2.3, aspect="auto")
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            v = m.iloc[i, j]
            ax.text(j, i, f"{v:.0f}" if v >= 100 else f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(z.iloc[i, j]) > 1.5 else INK)
    ax.set_xticks(range(len(models)), models)
    ax.set_yticks(range(len(PROFILE_FEATURES)), [FEATURE_LABELS[f] for f in PROFILE_FEATURES])
    ax.xaxis.tick_top()
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("z-score across the six models", color=INK2, fontsize=8)
    cb.outline.set_visible(False)
    save(fig, "rq4_feature_profiles")


def fig_feature_importance(cfg):
    imp = pd.read_csv(RESULTS / "rq4" / "feature_importance.csv").head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.barh([FEATURE_LABELS[f] for f in imp["feature"]], imp["permutation_importance"], height=0.55, color="#2a78d6")
    ax.set_xlabel("Drop in validation accuracy when the feature is shuffled")
    ax.grid(axis="y", visible=False)
    save(fig, "rq4_feature_importance")


def fig_ablations(cfg):
    a = pd.read_csv(RESULTS / "rq4" / "ablations.csv")
    order = [v for v in VARIANT_LABELS if v in set(a["variant"])][::-1]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for arch in ("cnn", "lstm", "tfidf_lr"):
        sub = a[a["arch"] == arch].set_index("variant")
        present = [v for v in order if v in sub.index]
        if len(present) < 2:
            continue
        ax.scatter([sub.loc[v, "test_accuracy"] for v in present], [order.index(v) for v in present], s=38,
                   color=ARCH_COLORS[arch], edgecolor="white", linewidth=1, label=ARCH_LABELS[arch], zorder=3)
    ax.set_yticks(range(len(order)), [VARIANT_LABELS[v] for v in order])
    ax.set_xlabel("Test accuracy after retraining on the modified responses")
    ax.grid(axis="y", color=GRID)
    ax.legend(loc="lower left", ncol=3, bbox_to_anchor=(0, 1.0))
    save(fig, "rq4_ablations")


def main():
    cfg = load_config()
    style()
    for fn in (fig_lengths, fig_training_curves, fig_confusion, fig_rq2, fig_rq3, fig_feature_profiles,
               fig_feature_importance, fig_ablations):
        try:
            fn(cfg)
            print("ok ", fn.__name__)
        except FileNotFoundError as e:
            print("skip", fn.__name__, "-", e)


if __name__ == "__main__":
    main()
