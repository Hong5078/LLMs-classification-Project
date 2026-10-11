# Project report

[Project_Report.pdf](Project_Report.pdf) is the submission report.
[report.tex](report.tex) is its editable LaTeX source; `generated/` contains tables and
number macros generated from the saved experiment results.

To rebuild the PDF without retraining, from the repository root run:

```bash
cd report
pdflatex -jobname=Project_Report -interaction=nonstopmode -halt-on-error report.tex
pdflatex -jobname=Project_Report -interaction=nonstopmode -halt-on-error report.tex
```

The repository is [Hong5078/LLMs-classification-Project](https://github.com/Hong5078/LLMs-classification-Project).
