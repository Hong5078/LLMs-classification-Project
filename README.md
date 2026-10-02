# Who Wrote It? Identifying LLMs from Their Responses

CMPSC 448 course project. Team leader and only member: Hongyu Guo.

Given a response written by a large language model, predict which of six LLM families wrote it
(ChatGPT, Gemini, Claude, Llama, DeepSeek, Qwen). A Kim-style text CNN and a bidirectional LSTM are
implemented in PyTorch and trained from scratch, with a TF-IDF + logistic regression model as a reference.

The report is [report/report.pdf](report/report.pdf). Plain-language study notes are in [NOTES.md](NOTES.md).

## Research questions

| | Question | Script |
|---|---|---|
| RQ1 | Can the LLM be identified from its response? | `src/exp_rq1.py` |
| RQ2 | Does the user's prompt help (input only / output only / input + output)? | `src/exp_rq2.py` |
| RQ3 | Do fingerprints transfer across question types? | `src/exp_rq3.py` |
| RQ4 | What distinguishes the models (stylometry, n-grams, ablations)? | `src/exp_rq4_features.py`, `src/exp_rq4_ablations.py` |

## Data

The data is **TXD-22: A Large-Scale Benchmark Dataset for Multi-Class AI-Generated and Mixed-Source
Text Identification** (Iqbal and Kashem, Mendeley Data, version 1, 2026, CC BY 4.0),
<https://data.mendeley.com/datasets/prcjcggtjf/1>, DOI `10.17632/prcjcggtjf.1`.

The CSV has 66,000 rows in 22 classes (3,000 each) and three columns: `Question`, `Answer`, `Source`.
The dataset is not committed to this repository. To get it:

1. Download the archive from the Mendeley page above and unzip it.
2. Put the CSV in `data/raw/` (any file name ending in `.csv`).

`python -m src.prepare` then builds the six-model subset:

- reads the file as Windows-1252 (it is not UTF-8),
- keeps the rows whose `Source` is one of `ChatGPT`, `Gemini`, `Claude`, `Llama`, `DeepSeek`, `Qwen`,
- drops empty responses, spreadsheet-error placeholders and exact duplicate responses (12 rows in total),
- labels every prompt with a question type (definitional / explanatory / mixed / other) using the rules in
  `src/prepare.py`,
- makes a 70/15/15 train/validation/test split, stratified by model and grouped by prompt, so that no prompt
  appears in more than one split (seed 42).

The subset is written to `data/processed/subset.csv` (not committed). The split assignment of every row is
committed in `results/splits.csv`, keyed by the row number in the original CSV.

## Setup

Python 3.10. A CUDA GPU is optional but recommended (all training runs took about two hours on a laptop RTX 4060).

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121   # or the CPU build
pip install -r requirements.txt
```

Compiling the report needs a LaTeX distribution with `pdflatex`.

## Reproducing everything

```bash
bash run_all.sh                                   # uses `python` on the PATH
PYTHON=.venv/Scripts/python.exe bash run_all.sh   # or name the interpreter (Windows venv shown)
```

This runs the steps below in order. Each step can also be run on its own with `python -m <module>`.
Finished training runs are cached as JSON in `results/`; delete a file (or the whole folder) to retrain.

| Step | Module | Output |
|---|---|---|
| Subset, cleaning, split | `src.prepare` | `data/processed/subset.csv`, `results/splits.csv`, `results/audit/cleaning_log.json` |
| Data audit | `src.audit` | `results/audit/` |
| One-epoch timing, max length | `src.exp_timing` | `results/timing.json` |
| RQ1 and validation tuning | `src.exp_rq1` | `results/tuning/`, `results/rq1/` |
| Error analysis | `src.error_analysis` | `results/rq1/error_*.{csv,json}` |
| RQ2 | `src.exp_rq2` | `results/rq2/` |
| RQ3 | `src.exp_rq3` | `results/rq3/` |
| RQ4 features and n-grams | `src.exp_rq4_features` | `results/rq4/` |
| RQ4 ablations | `src.exp_rq4_ablations` | `results/rq4/ablations/`, `results/rq4/ablations.csv` |
| Figures | `src.make_figures` | `figures/` |
| Tables and numbers for the report | `src.make_tables` | `report/generated/` |
| Report | `pdflatex report.tex` (twice, in `report/`) | `report/report.pdf` |

### Which file produces which table or figure

| Report item | Source file(s) in `results/` | Drawn or formatted by |
|---|---|---|
| Table 1 (dataset) | `audit/cleaning_log.json`, `audit/length_stats.csv` | `make_tables.dataset_tables` |
| Table 2 (audit of shortcuts) | `audit/self_identification.csv`, `audit/typography_rates.csv` | `make_tables.dataset_tables` |
| Table 3 (tuning grid) | `tuning/tuning_summary.csv`, `rq1/best_config.json` | `make_tables.training_tables` |
| Table 4 (RQ1) | `rq1/summary.csv`, `rq1/baseline_output.json` | `make_tables.rq1_tables` |
| Table 5 (RQ2) | `rq2/summary.csv`, `rq2/input_only_breakdown.json` | `make_tables.rq2_tables` |
| Table 6 (question types) | `audit/qtype_counts.csv`, `audit/qtype_prompts_per_split.csv` | `make_tables.dataset_tables` |
| Tables 7-8 (RQ3) | `rq3/transfer.csv`, `rq3/runs.csv` | `make_tables.rq3_tables` |
| Table 9 (feature groups) | `rq4/feature_groups.csv`, `rq4/feature_classifier.json` | `make_tables.rq4_tables` |
| Table 10 (n-grams, closing phrases) | `rq4/distinctive_ngrams.csv`, `rq4/closing_phrases.csv` | `make_tables.rq4_tables` |
| Table 11 (ablations) | `rq4/ablations.csv` | `make_tables.rq4_tables` |
| Table 12 (error examples) | `rq1/error_examples.csv`, `rq1/error_stats.json` | `make_tables.rq1_tables` |
| Figure 1 (response lengths) | `audit/lengths_per_row.csv` | `make_figures.fig_lengths` |
| Figure 2 (training curves) | `rq1/cnn_seed42.json`, `rq1/lstm_seed42.json` | `make_figures.fig_training_curves` |
| Figure 3 (confusion matrices) | `rq1/cnn_mean_confusion.json`, `rq1/lstm_mean_confusion.json` | `make_figures.fig_confusion` |
| Figure 4 (stylometric profiles) | `rq4/feature_means.csv` | `make_figures.fig_feature_profiles` |

`figures/` also contains four figures that are not in the report: `rq2_conditions`, `rq3_transfer`,
`rq4_feature_importance` and `rq4_ablations`. They show the same numbers as Tables 5, 7-8, 9 and 11.

Every number in the report text is a LaTeX macro defined in `report/generated/numbers.tex`, which
`src.make_tables` writes from the result files. No number is typed by hand.

## Repository layout

```
configs/default.yaml   all hyperparameters and grids
src/                   data preparation, tokenizer, models, training, evaluation, experiments, figures, tables
results/               metrics, training histories, confusion matrices, split assignment (JSON/CSV)
figures/               figures used in the report (PDF and PNG)
report/                report.tex, generated tables, report.pdf
NOTES.md               study notes: design decisions and results in plain language
run_all.sh             one command that reproduces everything
```

## Use of AI tools

An LLM coding assistant (Claude) was used to write and debug the code and to draft the report, under the
author's direction, as the project description allows. All experiments were run locally and every reported number comes from the saved
result files in `results/`.
