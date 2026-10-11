#!/usr/bin/env bash
# Reproduce every experiment, table and figure in the report.
# Requires the TXD-22 CSV in data/raw/ (see README.md). Finished runs are cached in results/;
# delete results/ to retrain everything from scratch.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python}"

$PY -m src.prepare              # six-model subset, cleaning, question types, grouped 70/15/15 split
$PY -m src.audit                # data audit tables (results/audit)
$PY -m src.exp_timing           # one-epoch timing and max-length choice under the compute budget
$PY -m src.exp_rq1              # validation tuning, TF-IDF baseline, CNN and BiLSTM with three seeds
$PY -m src.error_analysis       # confusion pairs and example errors
$PY -m src.exp_rq2              # input-only / output-only / input+output
$PY -m src.exp_rq3              # transfer across question types
$PY -m src.exp_rq4_features     # stylometric features, features-only classifier, distinctive n-grams
$PY -m src.exp_rq4_ablations    # retraining with one kind of signal removed
$PY -m src.make_figures         # figures/
$PY -m src.make_tables          # report/generated/ (LaTeX tables and number macros)

cd report
pdflatex -jobname=Project_Report -interaction=nonstopmode -halt-on-error report.tex > /dev/null
pdflatex -jobname=Project_Report -interaction=nonstopmode -halt-on-error report.tex > /dev/null
echo "report/Project_Report.pdf"
