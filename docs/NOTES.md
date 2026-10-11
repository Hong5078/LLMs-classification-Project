# Study notes

Plain-language notes on every major decision and result, so I can explain the project without the report in
front of me. Numbers are copied from `report/generated/numbers.tex` (generated from `results/`). If the pipeline
is rerun, check that file again.

## The project in five sentences

1. Task: read a response and say which of six LLMs wrote it (ChatGPT, Gemini, Claude, Llama, DeepSeek, Qwen).
   Guessing gives 16.7%.
2. Data: 17,988 responses from the public TXD-22 dataset. Every model answered the same 3,000 academic questions.
3. Models: a text CNN and a bidirectional LSTM, both in PyTorch and trained from scratch, plus a TF-IDF +
   logistic regression baseline.
4. Main result: CNN 87.5% (± 0.8), BiLSTM 84.6% (± 1.4), baseline 92.5% test accuracy. The simple baseline wins.
5. Why it works: mostly word choice and phrasing, with some help from punctuation. Not markdown, not model
   names, not curly quotes. The prompt is useless because all models got the same prompts.

## Data decisions

**Why six of the 22 classes?** The project asks for LLM families. The six I kept are the families the
assignment itself lists as examples, each from a different developer, and each has exactly one label in the
file, so no merging was needed. The other classes are human or mixed text (not LLMs), products that can route to
several underlying models (Poe, Perplexity), or a second model from a developer already included (Gemma).

**What cleaning was done?** Only 12 rows were removed: one Llama row whose "response" was the Excel error text
`#NAME?`, and 11 exact duplicate responses (8 Gemini, 3 DeepSeek). I did not lowercase or strip punctuation,
because those might be part of the fingerprint. Whether they are is tested later with ablations.

**Why is the file read as Windows-1252?** It is not valid UTF-8. Read as Latin-1, curly quotes turn into
invisible control characters. Read as Windows-1252 they become the correct curly quotes and dashes.

**What did the audit find?**
- Lengths: median 301 tokens, 99% of responses at most 504 tokens. Llama is shortest (244 tokens on average),
  Claude longest (344).
- Prompt overlap: 2,974 of the 3,000 prompt groups were answered by all six models. This is the key fact about
  the dataset.
- Giveaways: models almost never name themselves (at most 0.07% of a model's responses).
- Formatting: almost no newlines or markdown (the dataset authors cleaned whitespace). So the "markdown
  fingerprint" I expected is not in this data.
- Typography: curly apostrophes appear in 47% of ChatGPT, 52% of DeepSeek and 40% of Qwen responses, and almost
  never in Claude, Llama or Gemini. This could come from the models or from how answers were copied into the
  file. I cannot tell which, so I tested it with an ablation.
- Prompt encoding: questions with an apostrophe are stored with a real curly apostrophe in Qwen's rows and with
  a garbled sequence (`womenâ€™s`) in the other models' rows. 65 prompt strings therefore belong only to Qwen.

## Why split by prompt group?

Because the same question appears six times, once per model. If rows were split at random, the test set would
contain questions the classifier already saw in training (answered by other models, or even by the same
model's five neighbours). The test would then partly measure memory of topics. Splitting by group puts all six
answers to a question in the same split, so the test questions are completely new. I used
`StratifiedGroupKFold` with 20 folds: 14 for training (70%), 3 for validation (15%), 3 for testing (15%). Result:
12,590 / 2,699 / 2,699 rows from 2,100 / 450 / 450 prompts, with zero prompts shared.

**The mistake I made here.** My first grouping key compared the question text after lowercasing and removing
punctuation. It treated Qwen's curly-apostrophe version and the others' garbled version as two different
questions, so a few dozen questions were in different splits for Qwen. I caught it because an input-only CNN scored
19.9%, which should be impossible when everyone gets the same prompts. The fix was to delete non-ASCII
characters and apostrophes before comparing. I rebuilt the split and reran everything. Lesson: a result that is
better than it should be is a leakage warning.

## Why this tokenizer?

One regular expression: a newline, a run of letters, a run of digits, or any other single character. Case is
kept, and every punctuation mark is its own token.
- Punctuation and typography differ between models, so I wanted the networks to see them.
- Because the apostrophe is a separate token, `don't` and `don’t` share the word pieces and differ only in the
  apostrophe token. That makes the typography ablation clean.
- No pretrained tokenizer or embedding is involved, so the models really are trained from scratch.
- Vocabulary: built on the training split only (otherwise test words leak into the model), words seen at least
  twice, capped at 30,000, giving 30,005 entries with the special tokens.
- Max length 512 tokens covers 99.2% of training responses fully (384 would cover 86.4%, 256 only 30.0%).

## The two networks

**CNN (Kim 2014).** Embedding (200 dimensions) → three parallel 1D convolutions with kernel sizes 3, 4, 5 and
256 filters each → ReLU → max over time → concatenate (768 numbers) → dropout 0.3 → linear layer to 6 classes.
6.6 million parameters, almost all in the embedding table. Intuition: each filter is a detector for a 3- to
5-token pattern, and max-pooling asks "did this pattern occur anywhere?".

**BiLSTM (Hochreiter and Schmidhuber 1997).** Embedding (200) → dropout → 2-layer bidirectional LSTM, hidden
size 256 per direction → max-pool and mean-pool over the real tokens → concatenate (1,024 numbers) → dropout 0.5
→ linear layer. 8.5 million parameters. Packed sequences mean the LSTM skips padding. Pooling over all time
steps is used instead of the last hidden state because evidence can be anywhere in a 300-token answer.

**Training (both).** AdamW, learning rate 0.001, weight decay 0.01, batch size 64, cross-entropy loss, gradient
clipping at 1.0 for the LSTM, early stopping on validation macro-F1 with patience 3, at most 15 epochs. Seeds are
fixed. Batches are grouped by length (shuffle, sort inside big chunks, shuffle the batches), which halved the
LSTM's epoch time.

**Tuning.** A small grid, validation only, seed 42. CNN: filters {128, 256} × dropout {0.3, 0.5}; best was 256
filters, dropout 0.3 (validation macro-F1 87.3%). BiLSTM: hidden {128, 256} × layers {1, 2}; best was hidden
256 with 2 layers (84.0%). The differences inside the grid are about as large as differences between seeds, so
the choice is not very meaningful.

**Compute.** One epoch takes about 5 seconds for the CNN and 18 for the default BiLSTM on a laptop RTX 4060.
All 57 runs together took 115 minutes of training (79 for the networks, 36 for the baselines).

## RQ1 results and how to read them

| Model | Accuracy | Macro-F1 |
|---|---|---|
| CNN (3 seeds) | 87.5 ± 0.8 | 87.5 ± 0.8 |
| BiLSTM (3 seeds) | 84.6 ± 1.4 | 84.4 ± 1.6 |
| TF-IDF + LR | 92.5 | 92.4 |
| Chance | 16.7 | |

- Easiest: Claude and Llama (F1 around 92 to 97). Hardest: ChatGPT, Qwen and DeepSeek (F1 around 77 to 82).
- **Reading the confusion matrix:** each row is the true model. Each cell is the share of that model's test
  responses that got the column's label, so every row sums to 1 and the diagonal is recall. A bright
  off-diagonal cell means "this row model is often mistaken for this column model". The brightest ones are
  Qwen → ChatGPT (9.0% for the CNN) and ChatGPT → Qwen (8.2% CNN, 11.8% BiLSTM).
- Why those pairs: ChatGPT, Qwen and DeepSeek share curly-quote typography and have similar sentence lengths.
  Claude (very long sentences, 40 words on average) and Llama (short answers, repetitive vocabulary) are unlike
  everyone else.
- **Why the baseline wins:** the networks overfit (CNN training accuracy 93.9% vs validation 87.3% at the chosen
  epoch) because they learn 30,000 embeddings from only 12,590 examples. The signal is largely "how often does
  this model use this word or punctuation mark", which TF-IDF counts directly.

## RQ2 results

| Model | Input only | Input only (ASCII prompt) | Output only | Input + output |
|---|---|---|---|---|
| CNN | 16.8 | 16.5 | 86.6 | 87.5 |
| BiLSTM | 16.7 | 16.7 | 83.3 | 82.3 |
| TF-IDF + LR | 16.9 | 16.7 | 92.5 | 92.6 |

- **Prompt alone:** chance. Every prompt appears with all six labels, so one prediction per prompt can be right
  for only one of them.
- **Adding the prompt to the response:** no real change (+0.9, −1.1, +0.1 points, inside seed noise).
- **Could input-only ever work?** Yes, if the dataset leaks. Here the only leak is the Qwen apostrophe encoding.
  Of the 57 test rows whose prompt string is not shared by all six models, 10 are Qwen's; the baseline gets all
  10 right and the CNN 9. On the 2,642 rows with shared prompts accuracy is exactly chance. Removing non-ASCII
  characters from the prompts removes the cue.
- The general point: when input-only works, the classifier has learned how the dataset was assembled (different
  prompt sources, topics or encodings per model), not anything about the models.

## RQ3 results

TXD-22 has no coding, maths or writing tasks, so I could not do the split the assignment suggests. I used the
form of the question as the domain:
- *definitional*: starts with what / define / who / which / name / state ... and contains no explanatory word
  (2,093 prompts' worth per model);
- *explanatory*: starts with explain / how / describe / discuss / analyze / why / compare ... (786 per model).

Train on one type, test on the other, both directions. The definitional training set is subsampled to the size
of the explanatory one (about 3,478 rows each) so the comparison is fair.

| Model | Test type | Same-type training | Other-type training | Drop |
|---|---|---|---|---|
| CNN | definitional | 81.8 | 80.4 | 1.5 |
| CNN | explanatory | 83.6 | 83.3 | 0.3 |
| BiLSTM | definitional | 76.4 | 75.9 | 0.6 |
| BiLSTM | explanatory | 80.7 | 80.2 | 0.5 |

- The drop is tiny (at most 1.5 points) and about the size of seed noise.
- Claude and Llama stay the most identifiable in both directions.
- So the fingerprint is general with respect to question form. **But** the two domains are very similar (same
  subjects, same answer format), so this is a weak test. I should say that plainly if asked.
- Accuracy here is lower than in RQ1 only because the training sets are about a quarter of the size.

## RQ4 results

**Stylometric features.** 38 numbers per response in five groups (length, lexical, punctuation, typography,
formatting). Gradient boosting on only these features gets 79.7% (logistic regression 72.3%). That is most of
the CNN's 87.5% without seeing any words.
- Most important features (permutation importance): type-token ratio, curly apostrophe rate, character count,
  parenthesis rate, MATTR.
- One group alone: length 59.0%, lexical 55.2%, punctuation 55.5%, typography 42.4%, formatting 20.1%.
- Removing any one group costs only 4 to 5 points: the groups overlap in what they tell the classifier.
- **TTR vs MATTR:** the type-token ratio (distinct words / all words) falls as a text gets longer, so it partly
  measures length. MATTR averages the ratio over sliding 50-word windows, so it does not depend on length.

**Distinctive n-grams** (log-odds ratio with a Dirichlet prior; topic words cancel because all models answered
the same questions): ChatGPT uses curly double quotes and "especially"; Gemini uses `$` (LaTeX), parentheses,
colons, semicolons; Claude uses "that / through / while / including" and ", creating"; Llama uses "can" and
"used to"; DeepSeek uses parentheses, em dashes and "e.g."; Qwen uses commas, "for example" and "fostering".
Closings: 8.7% of ChatGPT answers end with a sentence starting "In summary"; 6.1% of Llama answers with "By
understanding".

**Ablations** (retrain after removing one kind of signal; seed 42; differences under about 2 points are noise):

| Change to the responses | CNN | BiLSTM | Baseline | What it shows |
|---|---|---|---|---|
| none | 86.6 | 83.3 | 92.5 | reference |
| (a) markdown stripped | 87.1 | 82.6 | 92.1 | no effect: there is almost no markdown in the data |
| (b) whitespace normalised | 86.8 | 83.1 | 92.6 | no effect: there are almost no newlines |
| (c) lowercase, no punctuation | 80.9 | 77.3 | 90.4 | case and punctuation are worth about 6 points (2 for the baseline) |
| (d) first 128 tokens | 75.8 | 73.4 | 81.2 | less text, less evidence |
| (d) first 64 tokens | 64.4 | 60.8 | 68.3 | same, stronger |
| (e) model names masked | 85.5 | 83.4 | 92.4 | no effect: models do not name themselves |
| (f) typography normalised | 87.3 | 82.3 | 91.9 | no effect, even though it changes 36% of responses |
| (c)+(d) words only, first 64 | 58.0 | 56.4 | 64.7 | still 3.5 times chance from 64 bare words |

- **Structural or lexical?** Both, mostly lexical. Words alone give the CNN 80.9 of its 86.6.
- **Typography surprise:** it looked like the biggest giveaway in the audit, but the classifiers do not need it.
  A big difference in a summary statistic does not prove the classifier depends on it. Only an ablation shows
  that.
- **Weak point of the truncation ablation:** cutting to 64 tokens hides the length *and* throws away 80% of the
  text, so the drop mixes two effects. Length features alone give 59%, but removing them from the feature
  classifier costs only 4.6 points, so I think most of the drop is lost evidence.

**Error analysis.** The CNN makes 361 errors and the BiLSTM 450 on 2,699 test rows (seed 42); 216 are shared.
Misclassified responses are shorter (about 275 tokens vs 304). Typical error: a plain definitional opening with
none of the model's habitual markers, or a response that uses another model's habit (a Qwen answer opening with
a parenthesis is labelled DeepSeek).

## Limitations I should be ready to state

- One dataset, one prompt style, one answer format, formatting flattened before release.
- Model versions and generation settings are not documented; responses date from early 2025.
- The fingerprint may partly be the collection pipeline, not the model.
- RQ3 is a mild domain shift only.
- RQ2 to RQ4 use one seed.
- The BiLSTM sometimes hit the 15-epoch limit while still improving.

## Questions an instructor might ask

**Why does a bag-of-n-grams model beat your neural networks?**
Small data and no pretraining. The networks must learn 30,000 word vectors from 12,590 examples and they overfit.
The task depends on frequencies of words and punctuation, which TF-IDF represents directly, and logistic
regression with L2 regularisation generalises well from that. Pretrained embeddings or a fine-tuned transformer
would likely close the gap.

**How do you know the classifier is not just learning topics?**
All six models answered the same questions, so topics are balanced across classes, and the split is by prompt
group, so test topics never appear in training.

**Is 87% impressive?**
It is 5 times chance on unseen questions, but below the simple baseline. The honest summary is that the models
are clearly distinguishable in this dataset and that a CNN from scratch is not the best tool for it.

**Why macro-F1 as well as accuracy?**
The classes are balanced, so they are nearly equal here. Macro-F1 guards against a model that ignores a class,
and the per-class F1 scores show which models are hard.

**Why early stopping on validation macro-F1 and not on loss?**
The metric I report is F1, so I select on it directly. Validation loss is also noisier in my runs and can rise
while accuracy is still creeping up (the model becomes over-confident on its mistakes); for the BiLSTM the lowest
validation loss came two epochs before the best F1. Either choice is defensible.

**Why did you evaluate on the test set only once per model?**
Any choice made by looking at test results (hyperparameters, epochs, which ablation to report) would make the
test score optimistic. All choices used validation.

**What is the leak you found, in one sentence?**
Qwen's prompts with apostrophes are stored with a different character than the other models' prompts, so 65
prompt strings identify Qwen; it affected my first split and the input-only experiment, and I fixed it by
grouping on ASCII-normalised text.

**Would this work on ChatGPT output from today?**
Unknown, and probably worse. The data is from early 2025 with undocumented versions, and real chat output has
markdown that this dataset removed.

**Why 512 tokens?**
It fully covers 99.2% of training responses, and the timing check showed the whole pipeline fits the compute
budget at that length.

**Why are RQ3 numbers lower than RQ1?**
The RQ3 models are trained on about 3,478 rows instead of 12,590. With all 8,590 definitional rows the CNN
reaches 87.2% in-domain and 88.9% on explanatory questions.

**What would you do next?**
Pretrained representations, several seeds for every experiment, a dataset with real task variety and known model
versions, and a length-controlled ablation using random fixed-length windows.

## How the repository guarantees the numbers

Every training run writes a JSON file under `results/`. `src/make_tables.py` reads those files and writes the
LaTeX tables and a file of number macros (`report/generated/numbers.tex`). The report text uses the macros, so
no number is typed by hand. `bash run_all.sh` reruns everything in order.
