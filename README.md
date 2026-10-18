# CausalMoE: Causal-aware Adaptive Routing for Mixture of Experts

This repository is the cleaned implementation of the first work in the thesis:
**Causal-aware Adaptive Routing Mechanism for Mixture of Experts (CausalMoE)**.

Detailed notes for the first work (method, code map, experiment protocol, known
issues and thesis-writing caveats): [docs/FIRST_WORK.md](docs/FIRST_WORK.md).

It covers the paper's four experiment lines:

1. Language-model pretraining/evaluation on WikiText-103, WikiText-2, and PTB.
2. Code fine-tuning/evaluation on HumanEval, MBPP, APPS, and Codeforces.
3. Corr2Cause fine-tuning/evaluation.
4. CausalBench (Code/Math/Text) fine-tuning/evaluation.

The repository contains code and small configuration files only. Datasets, tokenizers,
checkpoints, and logs are expected to live outside git.

## Method in one paragraph

CausalMoE keeps the standard Top-K routers in the first `L-1` MoE layers. In the last
MoE layer it builds a block-level causal graph from a compressed attention branch,
uses CLEANN to obtain the explanation set of the most attended block, and routes all
tokens in the selected blocks to one additional Causal Expert. The general experts
still use Top-2 among the remaining 15 experts; the causal expert does not compete in
Top-K. The general and causal outputs are fused by a learned sigmoid gate. A Causal
Auxiliary Loss (CAL) penalizes the cosine similarity between the causal expert and the
general experts, and is applied only to the last layer.

## Repository layout

```
configs/
  models.json          21 model configurations (7 base + 7 causal w/o CAL + 7 causal + CAL)
  tasks.json           task-level defaults (epochs, batch size, data subdirectory)
scripts/
  run.py               parameterized launcher (recommended entry point)
  *.sh                 legacy paper-era scripts (absolute /hy-tmp paths; reference only)
scripts_causal/
  *.sh                 legacy causal scripts (reference only)
models_causal.py       CausalMoE model, all p/q/k and c/d/t layer variants
custom_layers_causal.py causal MoE layer, routing, CLEANN integration, CAL-adjacent code
custom_transformer_causal.py  causal transformer blocks
data.py                LM/code data loader + Corr2Cause/CausalBench loader
train_causal.py        dispatcher: chooses train_lm.py or train_reasoning.py from --data
train_lm.py            PPL/code fine-tuning entry point
train_reasoning.py     Corr2Cause/CausalBench entry point
trainer.py             trainer for LM/code tasks
trainer_reasoning.py   trainer for Corr2Cause/CausalBench
causalitylab/          bundled causal discovery/reasoning package
data_directory/        data preparation scripts and expected layout (no datasets)
```

## Environment

The model depends on the FastMoE CUDA extension (`fmoe_cuda`). On the GPU server:

```bash
pip install -r requirements.txt
pip install -e .          # builds the FastMoE CUDA extension; requires CUDA + ninja
```

If the CUDA build is not available, the import of `fmoe_cuda` will fail; this is expected.
The code was developed for a single 4090/24G GPU with PyTorch and CUDA.

## Data layout

The runner uses `--data-root data_directory` by default. Place each dataset as follows.

### Language/code datasets

`data.py` accepts `.txt`, `.parquet`, or `.json` splits:

```
data_directory/
  wikitext-103/
    train.txt | train.parquet
    valid.txt | validation.parquet
    test.txt  | test.parquet
  wikitext2/...
  ptb_text_only/...
  MBPP/...
  APPS/...
  HumanEval/...
  Codeforces/...
```

For Parquet, the loader auto-detects a text column among `text`, `solution`, `code`,
`content`, `question`, and `answer`; list-valued cells are joined with spaces. The
vocabulary is built from train/valid/test and saved to `my_custom_tokenizer/`. The
reasoning path reuses this tokenizer, so run at least one LM/code pretraining job (or
place a compatible tokenizer in the working directory) before causal-reasoning runs.

Small data-preparation scripts are kept under `data_directory/*/deal.py`; they document
how the paper-era datasets were converted. They are not required when your data is
already in the layout above.

### Corr2Cause / CausalBench

Both reasoning datasets use:

```
data_directory/
  corr2cause/
    train.json
    valid.json
    test/            # one or more JSON files, each a list of {"text": ..., "label": ...}
  causal-bench/
    train.json
    valid.json
    test/            # the 13 domain/perspective test files used in the thesis table
```

`train_reasoning.py` reads `train.json` and `valid.json` for training, and every JSON
file under `test/` for evaluation. The exact JSON schema is a list of objects with a
`text` field and a `label` field; `read_and_process_json_test` is the single place to
adapt if your files use different key names.

## Model manifest

`configs/models.json` contains the 21 configurations used in the thesis:

- SMoE family: `smoe-m`, `smoe-mom-m`, `smoe-adam-m`, `smoe-mom-l`.
- GLaM family: `glam-m`, `glam-mom-m`, `glam-adam-m`.
- Each non-large family has three entries: base, causal w/o CAL, causal + CAL.
  `smoe-mom-l` has base, causal w/o CAL, and causal + CAL.

Examples:

| id | display name | architecture |
|---|---|---|
| `smoe-m` | SMoE-medium | `sgsgsgsgsgsp` |
| `smoe-m-causal` | CausalSMoE-medium(w/o CAL) | `sgsgsgsgsgsc` |
| `smoe-m-causal-loss` | CausalSMoE-medium | `sgsgsgsgsgsc` |
| `smoe-mom-m-causal-loss` | CausalMomentumSMoE-medium | `smsmsmsmsmsd` |
| `smoe-mom-l-causal-loss` | CausalMomentumSMoE-large | `smsmsmsmsmsmsmsmsmsmsmsd` |
| `glam-m-causal-loss` | CausalGLaM-medium | `sgsfsgsfsgsfsgsfsgsfscsf` |

Run `python scripts/run.py --list` for all 21 entries. `c` = causal SMoE, `d` = causal
momentum, `t` = causal Adam; `p/q/k` are the corresponding non-causal layers that
return expert counts.

## Running experiments

Always start with a dry run to inspect the command:

```bash
python scripts/run.py --model smoe-m --task pretrain --dry-run
```

### 1. Pretraining / PPL

Pretrain on WikiText-103, then evaluate on WikiText-103/WikiText-2/PTB by pointing
`--data-root` at the corresponding directory or by editing `configs/tasks.json`:

```bash
python scripts/run.py --model smoe-m --task pretrain
python scripts/run.py --model smoe-m-causal-loss --task pretrain
python scripts/run.py --model smoe-m --task pretrain --eval
```

The checkpoint is written to `checkpoints/<model>/pretrain/smoe.pt`.

### 2. Code fine-tuning

```bash
python scripts/run.py --model smoe-m --task finetune_code --dataset MBPP
python scripts/run.py --model smoe-m-causal-loss --task finetune_code --dataset HumanEval
```

The pretrained checkpoint defaults to `checkpoints/<model>/pretrain/smoe.pt`; override
with `--pretrained-weight` if needed. The dataset name must match the directory under
`data_directory/` (case-sensitive).

### 3. Corr2Cause

```bash
python scripts/run.py --model smoe-m-causal-loss --task corr2cause
python scripts/run.py --model smoe-m-causal-loss --task corr2cause --eval
```

### 4. CausalBench

```bash
python scripts/run.py --model smoe-m-causal-loss --task causalbench
python scripts/run.py --model smoe-m-causal-loss --task causalbench --eval
```

For CAL models, the runner appends `--causal-loss`; for w/o CAL models it does not.
`--eval` appends `--resume --full-eval-mode`.

## Legacy scripts

`scripts/` and `scripts_causal/` are kept as the original paper-era command reference.
They contain absolute `/hy-tmp` paths and per-model `mkdir`/`CUDA_VISIBLE_DEVICES`
settings. Use them only as a record; prefer `scripts/run.py` for portable runs.

## Notes on the CAL implementation

The original prototype added the CAL value to the logging scalar (`loss_value`) but not
to the tensor passed to `backward()`, so the regularization had no gradient effect. In
this cleaned version the CAL term is added to the training loss before backward:

```python
loss = nll_loss(...)
if causal_loss:
    loss = loss + 0.01 * get_cosine_regularization(model)
```

The term is still computed only from the last layer's causal expert and the 15 general
experts, using the mean absolute cosine similarity of the `htoh4` and `h4toh` weights.
If you are comparing against historical CAL numbers, re-run the CAL ablation with this
version before drawing conclusions.

## Version mapping

- Paper main method (compression + 15 general + 1 causal + CAL): `8a2f37c` on the
  `deepseekmoe+loss` line, plus the evaluation code from
  `causal-bench-deepseekmoe+loss`.
- `DeepseekMoe` is a separate no-compression / percentage-token experimental line; it
  is not part of this main path.
- This cleaned branch intentionally omits datasets, checkpoints, IDE files, caches, and
  build artifacts.

## Verification

Static checks can be run without a GPU:

```bash
python -m py_compile *.py causalitylab/**/*.py
python scripts/run.py --list
python scripts/run.py --model smoe-m --task pretrain --dry-run
python scripts/run.py --model smoe-m-causal-loss --task causalbench --dry-run
```

Full training requires `fmoe_cuda`, CUDA, and the external datasets; it cannot be
validated on a CPU-only checkout.
