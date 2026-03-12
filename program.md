# Autonomous Research Agent Instructions

You are an autonomous AI research agent working on this repository. Your objective is to iteratively modify the codebase, run experiments, and improve the model's performance without requiring human intervention.

## 🎯 Objective & Evaluation
* **Evaluation Criterion:** Your primary metric for success is **GSM8K accuracy** as measured by `eval_math.py`. Your goal is to maximize this value.
* **Train Command:** To train a model, execute:
  `bash scripts/train_math.sh`
  Before each trial, set a unique `output_dir` in the config that reflects what the trial is testing. Use short, lowercase, hyphen-separated keywords derived from the trial description. Examples:
  - description `"increase kl_coef to 0.05"` → `output_dir: ./output/kl-coef-0.05`
  - description `"lr 1e-6 with cosine warmup"` → `output_dir: ./output/lr-1e6-cosine`
* **Eval Command:** After training completes, evaluate the saved checkpoint with:
  `python eval_math.py --model_name_or_path <output_dir>/final`
  Parse the accuracy from the output line: `Accuracy: XX.X%  (correct/total)`

## 🔄 The Research Loop
For every experiment, follow this continuous loop:
* **Set up:** Create a single branch for the entire session (all trials share it). Follow the date rule for naming:
  `git checkout -b experiment-$(date +%Y%m%d-%H%M%S)`
* **Baseline:** You do not need to re-run the baseline; it has already been evaluated as `python eval_math.py --model_name_or_path output/math_grpo_1.5b/final` with GSM8K accuracy **69.8%**.
* **Initialize `results.tsv`:**
  Create `results.tsv` file (tab-separated, NOT comma-separated — commas break in descriptions). The TSV has a header row and 4 columns: `commit    accuracy    status  description`:
    1. git commit hash (short, 7 chars)
    2. GSM8K accuracy achieved (e.g. 69.8%) — use NaN for crashes
    3. status: keep, discard, or crash
    4. short text description of what this trial tried
  Log the baseline to `results.tsv` file.
* **Hypothesize:** Propose a code or config modification that could potentially increase GSM8K accuracy or reduce training time without degrading accuracy.
  Do **not** modify the `model` or `data` fields of `configs/math_grpo_1.5b.yaml`; only change `grpo`, `rollout`, `training`, `infra`, and `logging.output_dir`.
* **Implement:** Write the code changes.
* **Train:** Set `output_dir` in the config to a descriptive path (see naming convention above), then run `bash scripts/train_math.sh` to launch a new trial.
  After the first 5 training steps of training, estimate how long 1 epoch will take; if it is projected to take more than 8 hours, stop the trial and discard this change because it is too time-consuming.
* **Evaluate:** Run `python eval_math.py --model_name_or_path <output_dir>/final` and parse the resulting accuracy.
* **Analyze:** Compare the new accuracy to the current highest accuracy.
* **Commit or Revert:** If accuracy improves, commit the changes to your experimental branch. If it degrades or errors out, discard the working tree changes with `git restore .` — do not commit.
* **Summarize:** Log it to `results.tsv`. Always log the trial even it's been discard so there is a record of what was tried.

## ⚠️ Core Directives

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

* **Keep the codebase clean:** Write readable, modular code. Always revert failed trials entirely. Remove temporary logging, debugging statements, or test files before finalizing an iteration. Do not leave the repository in a broken state.

* **Simplicity criterion:** All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 accuracy improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 accuracy improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

* **NEVER STOP:** Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working indefinitely until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

* **Never push to `main`:** **Under no circumstances should you commit, merge, or push to the `main` or `master` branch.** Always work on isolated experimental branches.
