# OldNewsBERT

`ccp-oldnewsbert-2024` is an experimental Danish masked language modeling project focused on pretraining and continued pretraining for historical and news-oriented text corpus. The project utilizes the `Endevældens Nyheder Online (ENO)` dataset. The repository contains two main training tracks:

1. an **earlier from-scratch BERT-style pipeline** built around a custom WordPiece tokenizer and a `BertForMaskedLM` model, and
2. a **V3 pipeline** that continues pretraining from `vesteinn/DanskBERT` on a full version of the ENO-dataset.

The codebase is structured as a practical research repository rather than a polished package. Most of the logic lives in standalone training scripts intended to run on UCloud or a comparable GPU environment mounted under `/work/Ccp-OldNewsBERT_2024`.

## Project objective

The repository is designed to support Danish language model development for older and domain-shifted textual material. In practice, that means:

- building dataset artifacts for masked language modeling,
- training or benchmarking custom tokenizers,
- tokenizing long-form text with truncation and stride,
- training BERT-style masked language models,
- resuming long-running checkpointed training jobs,
- evaluating checkpoints with loss and perplexity,
- and exporting finished artifacts to the Hugging Face Hub.

The project have evolved over time. Therefore, the V3 workflow is the most coherent training track in the repository and should be treated as the current baseline. The V1 + V2 are mostly experimental and part of a learning-proces and based on a subset of the ENO-dataset.

## Technical overview

### Core stack

The repository targets **Python 3.12+** and depends primarily on:

- `torch`
- `transformers`
- `datasets`
- `tokenizers`
- `pandas`
- `numpy`
- `dask[complete]`
- `scikit-learn`
- `sentencepiece`

Package management is defined through and `requirements.txt`, which suggests mixed usage between Poetry-style local development and shell-based cluster execution.

### Modeling paradigm

All training code is based on **masked language modeling (MLM)** using Hugging Face Transformers. The repository contains two model strategies:

#### 1. From-scratch BERT-style training

The legacy pipeline:

- builds a tokenizer from local corpus text,
- creates train/test splits,
- tokenizes with optional truncation and sliding windows,
- instantiates a fresh `BertForMaskedLM` from `BertConfig`,
- trains with `Trainer` and `DataCollatorForLanguageModeling`.

#### 2. Continued pretraining from DanskBERT

The V3 pipeline:

- loads the `JohanHeinsen/ENO` dataset from Hugging Face,
- filters rows where `pwa >= 0.9`,
- trains a custom WordPiece tokenizer,
- initializes the model from `vesteinn/DanskBERT`,
- continues MLM training on the filtered corpus,
- evaluates checkpoints using perplexity derived from validation loss

The V3 model was firstly trained on the Lumi-HPC system, but as project-credits ran out the project was migrated back to Ucloud.

This second approach is operationally cleaner and the newest pipeline. The V3 is also the model that should be used, if working with the model is intended.

## Repository structure

```text
.
├── requirements.txt
├── setup.sh
├── modules/
│   └── OldNews_fun.ipy #Older functions for V1+V2 setup
├── py-scripts/
│   ├── 01_datastructure.py
│   ├── 02_tokenizer.py
│   ├── 03_apply_tokenizer.py
│   ├── 04_trainer.py
│   ├── 05_trainer.py
│   ├── V3/ # Main project folder
│   │   ├── 01_toknizer.py
│   │   ├── 02_train.py
│   │   ├── 03_evaluate.py
│   │   ├── 04_eval_checkpoint.py
│   │   ├── 05_train_checkpoint.py
│   │   ├── 06_train_checkpoint_extend.py
│   │   ├── 07_upload.py
│   │   ├── trainer.sh
│   │   ├── evaluate.sh
│   │   └── evaluate_earlier.sh
│   └── validation/
│       ├── pick_samples.py
│       ├── pseudo_ppl.py
│       └── calculations.py
├── runners/ # Bash-script for training.
│   ├── trainer.sh
│   └── trainer_multi.sh
├── modelling/
│   └── tokenizer/
├── tokenizer/
│   └── V3/
└── tokenizer_dir/
```

## Workflow

## 1. Environment setup

For UCloud-style execution, the repository expects shell-based installation through `setup.sh`:

```bash
bash setup.sh
```

This upgrades `pip` and installs dependencies from:

```bash
/work/Ccp-OldNewsBERT_2024/requirements.txt
```

Because the path is hardcoded, the script assumes the project is mounted exactly at `/work/Ccp-OldNewsBERT_2024`. If you are running elsewhere, update the paths before execution.

For local development:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
Or for local uv development:
```bash
uv venv
uv pip install -r requirements.txt
```

## 2. Legacy pipeline: local corpus to BERT MLM

The older pipeline is built around a local CSV dataset named `ONBert_data.csv`. This is an older ENO-dataset version and only a subset of the full 1.0 version available on `HuggingFace`.

### Step 1: build the dataset artifact from the older version.
*Note* This subset, as mentioned, is smaller and not as clean as the official version available. Proceed only as inspiration for working with your own, similar dataset.
```bash
python py-scripts/01_datastructure.py
```

This script:

- reads `data/ONBert_data.csv` using `pyarrow.csv` with `;` as delimiter,
- converts the file to Parquet,
- loads it through Dask,
- drops `lignende_tekster` and `kontekst`,
- removes short rows (`text_length < 30`),
- strips whitespace from the `text` field,
- converts the result into a Hugging Face `Dataset`,
- performs a 90/10 train-test split,
- saves the dataset to `data/dataset`.

### Step 2: train tokenizers

```bash
python py-scripts/02_tokenizer.py
```

This script trains two tokenizer variants on the training split:

- **ByteLevelBPETokenizer**
- **BertWordPieceTokenizer**


This is due to experimenting. The BertWordPieceTokenizer is the one used for the model to allign with the rest of the BERT-workflow.

It also:

- saves tokenizer files under `modelling/tokenizer/`,
- writes a manual `config.json` for the WordPiece tokenizer,
- benchmarks tokenization on random test samples,
- compares vocabulary sizes,
- computes an approximate OOV rate on a test subset.

### Step 3: apply tokenizer and build tokenized dataset

```bash
python py-scripts/03_apply_tokenizer.py
```

This script loads the saved WordPiece tokenizer and tokenizes the dataset into model-ready tensors.

Implementation details:

- token length statistics are computed first without truncation,
- the main tokenization path uses `max_length=512`,
- sliding-window chunking is configured with `stride=256`,
- output fields include `input_ids`, `attention_mask`, and `special_tokens_mask`,
- the final artifact is saved to `data/dataset_tokenized`.

A note of caution: this script contains repeated tokenization blocks and appears to preserve only one overflow chunk per example in its final section. It is usable as an experiment script, but it should be reviewed before being treated as production preprocessing.

### Step 4: train a BERT MLM model

Single-process variant:

```bash
python py-scripts/04_trainer.py
```

Distributed-oriented variant:

```bash
python py-scripts/05_trainer.py
```

The training configuration includes:

- `BertForMaskedLM`
- vocabulary size `30_522`
- sequence length `512`
- batch size `16`
- gradient accumulation `4`
- cosine LR schedule
- weight decay `0.01`
- checkpoint saving during training
- mixed precision when CUDA is available

The legacy trainers split the training data again to derive a validation set and optimize against `eval_loss`.

### Step 5: launch through helper runners

Single GPU / default launch:

```bash
bash runners/trainer.sh
```

Multi-GPU launch:

```bash
bash runners/trainer_multi.sh
```
Utilizing a Multi-GPU launch is highly recommnded in pipeline. Always check for GPU-availability on your system. 

## 3. V3 pipeline: Fine-tuning of `vesteinn/DanskBERT`

The V3 directory is the most complete end-to-end workflow in the repository.
This directory is newest model, also presented at the `DHNB 2026`. The workflow is an experiment in fine-tuning of the DanskBERT-model on the full ENO-dataset in order to get better results with more data, a baseline model and custom tokenizer. The results are faster training-schedule with a boost in performance.

### Dataset source

The V3 scripts use:

```text
JohanHeinsen/ENO
```

loaded via Hugging Face Datasets.

The preprocessing logic is consistent across training and evaluation:

- load dataset,
- filter rows where `pwa >= 0.9`,
- shuffle with fixed seeds,
- split train/test,
- tokenize with truncation and overlapping windows.

### Step 1: train the V3 tokenizer

```bash
python py-scripts/V3/01_toknizer.py
```

This script trains a custom `BertWordPieceTokenizer` on the filtered ENO training split and saves it under:

```text
/work/Ccp-OldNewsBERT_2024/tokenizer/V3
```

Tokenizer configuration:

- vocab size: `30_512`
- min frequency: `2`
- special tokens:
  - `[PAD]`
  - `[UNK]`
  - `[CLS]`
  - `[SEP]`
  - `[MASK]`
  - `<S>`
  - `<T>`
- `model_max_length=512`

### Step 2: train V3 from DanskBERT

```bash
python py-scripts/V3/02_train.py
```

This is continued pretraining, not pure from-scratch training. The script:

- loads the custom V3 tokenizer,
- tokenizes the ENO dataset with:
  - `max_length=512`
  - `stride=256`
  - no static padding at preprocessing time,
- initializes the model from `vesteinn/DanskBERT`,
- performs MLM training with dynamic padding via the data collator,
- saves checkpoints under:

```text
/work/Ccp-OldNewsBERT_2024/modelling/V3/models/ON-BERT-v3
```

Key training parameters:

- epochs: `5`
- per-device train batch size: `16`
- gradient accumulation: `4`
- per-device eval batch size: `32`
- learning rate: `5e-5`
- warmup ratio: `0.06`
- scheduler: `cosine`
- weight decay: `0.01`
- `mlm_probability=0.15`
- `pad_to_multiple_of=8`

### Step 3: resume checkpoint training

There are two continuation scripts:

```bash
python py-scripts/V3/05_train_checkpoint.py
python py-scripts/V3/06_train_checkpoint_extend.py
```

These are intended for extending long-running jobs from saved checkpoints, with explicit resume paths such as:

- `checkpoint-57020`
- `checkpoint-98640`

The extended run pushes training to `max_steps=120000` in the latest continuation script.

### Step 4: evaluate

Latest-model evaluation:

```bash
python py-scripts/V3/03_evaluate.py
```

Earlier-checkpoint evaluation:

```bash
python py-scripts/V3/04_eval_checkpoint.py
```

These evaluation scripts:

- load a checkpoint from disk,
- rebuild the filtered ENO validation pipeline,
- evaluate using Hugging Face `Trainer`,
- derive perplexity as `exp(eval_loss)`,
- write JSON output under `output/eval_results/`.

Shell wrappers are also included:

```bash
bash py-scripts/V3/trainer.sh
bash py-scripts/V3/evaluate.sh
bash py-scripts/V3/evaluate_earlier.sh
```

### Step 5: upload artifacts to Hugging Face

```bash
python py-scripts/V3/07_upload.py
```

This script:

- loads a trained checkpoint,
- saves the tokenizer into the model directory,
- authenticates with the Hugging Face Hub,
- creates a public repository,
- uploads the final folder contents.

The current upload target is configured for:

```text
SirMappel/DA-BERT_Old_News_V3
```

The model is later moved to:

```text
CALDISS-AAU/DA-BERT_Old_News_V3
```

You will need a valid `HF_TOKEN` in the environment, or an interactive login, before upload will succeed.

## Validation utilities
This workflow is a testing pipeling on unseen, hold-out data. It utilizes two subsets datasets from the same period. One, Den Vestsjællandske Avis being both temporally and contextually alligned with the training data and the second, a N.F.S Grundtvig filtered subset, being only temporally alligned.

The `py-scripts/validation/` directory contains auxiliary evaluation code for pseudo-perplexity experiments on held-out text samples.

### `pick_samples.py`

Creates fixed random samples from validation corpora such as:

- `Unseen_text.csv` -  Represents the `Den Vestsjællandske Avis`-sample
- `Grundtvig_sample.csv` - Represents the `N.F.S Grundtvig`-sample

and writes them to JSON for reproducible evaluation.

### `pseudo_ppl.py`

Computes pseudo-perplexity by iteratively masking tokens and scoring them with the trained MLM. The script currently targets:

```text
FacebookAI/xlm-roberta-base
```

This is generally useful for cross-model comparisons on unseen domain data.
Several models where tested for this workflow and changed iteratively. A function or Class for changing the input model would be more ideal for this workflow, considering several models was tested.

The full list of models tested:
```bash
CALDISS-AAU/DA-BERT_Old_News_V1
CALDISS-AAU/DA-BERT_Old_News_V3
CALDISS-AAU/DA-BERT_Old_News_V3 - earlier training checkpoint
FacebookAI/xlm-roberta-base
MiMe-MeMo/MeMo-BERT-03
```

### `calculations.py`

Loads stored evaluation results, extracts perplexity values, and exports summary statistics to CSV.

## Important implementation notes

This repository is research-driven and experimental in nature. It does contain a few rough edges that are worth understanding before reuse.

### Hardcoded paths

Most scripts assume the project exists at:

```text
/work/Ccp-OldNewsBERT_2024
```

If your environment differs, many scripts will fail without path edits.

### Script quality varies

Some scripts are stable enough for direct execution, while others show signs of iterative experimentation:

- duplicated tokenization logic,
- inconsistent directory naming (`modelling/tokenizer` vs `tokenizer/V3` vs `tokenizer_dir`),
- multiple trainer variants with partially overlapping responsibilities,
- shell wrappers that reference filenames not present in the listing.

In particular, `py-scripts/03_apply_tokenizer.py` should be reviewed before relying on it in a fresh pipeline. This was written mainly for testing the tokenizer types and thus experiment with the different workflows. This might be reviewed in the future.

### Naming inconsistencies

There are minor inconsistencies such as:

- `01_toknizer.py` instead of `01_tokenizer.py`,
- `modelling/tokenizer` versus top-level `tokenizer/`,
- V3 shell wrappers referencing evaluation files that may have been renamed.

These are not hard blockers, but they do matter when automating the workflow.
Please consult these as refernce or inspiration rather than autmatic workflows.

### Tokenizer strategy

The repository contains multiple tokenizer artifacts:

- a legacy WordPiece tokenizer,
- a ByteLevel BPE tokenizer,
- a V3 WordPiece tokenizer,
- a separate `tokenizer_dir/` with RoBERTa-style configuration and Danish character special tokens.

Only part of this is wired into the active training scripts. The V3 pipeline uses the tokenizer stored in `/work/Ccp-OldNewsBERT_2024/tokenizer/V3`.

## Suggested execution order

If you want to reproduce the current training direction, the most sensible order is:

```bash
bash setup.sh
python py-scripts/V3/01_toknizer.py
python py-scripts/V3/02_train.py
python py-scripts/V3/03_evaluate.py
python py-scripts/V3/07_upload.py
```
Also reference the V3-pipeline if fine-tuning is the primary objective. As mentioned this workflow is generally cleaner and better executed. Learning is part of this project.

If you want the older fully custom pipeline instead with a focus on pre-training a model from scratch see the older files:

```bash
bash setup.sh
python py-scripts/01_datastructure.py
python py-scripts/02_tokenizer.py
python py-scripts/03_apply_tokenizer.py
python py-scripts/04_trainer.py
```

## Example commands

### Local training

```bash
python py-scripts/V3/02_train.py
```

### Multi-GPU training through Accelerate

```bash
bash py-scripts/V3/trainer.sh
```

### Multi-GPU training through torchrun

```bash
bash runners/trainer_multi.sh
```

### Evaluate a saved checkpoint

```bash
python py-scripts/V3/04_eval_checkpoint.py
```

## Output artifacts

Depending on the path taken, the repository produces:

- serialized Hugging Face datasets in `data/`
- tokenizer vocab/config files in `modelling/tokenizer/` or `tokenizer/V3/`
- model checkpoints in `modelling/V3/models/ON-BERT-v3/`
- evaluation JSON files in `output/eval_results/`
- validation statistics CSV files in `output/ppl_stats_results/`

## Reproducibility

The scripts use fixed seeds in several places, including:

- `666`
- `420`
- `42`

This improves reproducibility for dataset shuffling and splitting, but full determinism is not guaranteed because training depends on GPU execution, masking, checkpoint state, and distributed runtime behavior.

## Known limitations

- The repository is not packaged as a reusable library.
- Paths are tightly coupled to a specific filesystem layout. Specifically the HPC-system used during development.
- Some scripts mix experimentation and production concerns.
- Data availability is assumed rather than managed internally.
- There is no top-level orchestration script for the full workflow.
- There is no formal test suite.

## Recommended next cleanup steps

Before using this codebase as a long-term training foundation, the following refactors would have high value:

1. centralize path configuration in a single settings module,
2. unify tokenizer storage conventions,
3. remove dead or duplicated tokenization code,
4. split preprocessing, training, evaluation, and upload into explicit CLI entrypoints,
5. pin exact dependency versions for cluster reproducibility,
6. add dataset provenance documentation,
7. add experiment metadata logging and checkpoint manifests.

## Authors

**CALDISS**  
`caldiss@adm.aau.dk`

## License

No license is currently specified in the repository metadata. Treat the project as internal or all-rights-reserved until a license is added explicitly.

