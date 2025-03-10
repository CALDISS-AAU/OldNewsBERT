# Packages
from pathlib import Path
import os
from transformers import AutoConfig
from tokenizers import BertWordPieceTokenizer
from datasets import Dataset
from datasets import *
import pandas as pd
import json
import sys
from os.path import join
import random
import pyarrow.csv as pv
import pyarrow.parquet as pq
import dask.dataframe as dd

# DIRS AND PATHS
project_dir = join('/work', 'Ccp-OldNewsBERT_2024')
data_dir = join(project_dir, 'data')
modelling_dir = join(project_dir, 'modelling')
os.makedirs(modelling_dir, exist_ok=True)
tokenizer_dir = join(modelling_dir, 'tokenizer')
os.makedirs(tokenizer_dir, exist_ok=True)
bpe_dir = join(tokenizer_dir, 'bpe_tokenizer')
wordpiece_dir = join(tokenizer_dir, 'wordpiece_tokenizer')
modules_dir = join(project_dir, 'modules')
sys.path.append(modules_dir)

logs_dir = join(modelling_dir, 'logs')
output_dir = join(project_dir, 'output')
model_dir = join(modelling_dir, 'models')

# data
dataset = load_from_disk('/work/Ccp-OldNewsBERT_2024/data/dataset')
print(dataset)
dataset['train'], dataset['test']

# Setting up tokenizers training schedule
# bpe_tokenizer
bpe_tokenizer = ByteLevelBPETokenizer()

bpe_tokenizer.train_from_iterator(
    dataset["train"]["text"],  # Train on text column
    vocab_size=30_512,
    min_frequency=2,
    special_tokens = [
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"
        ]
)
bpe_tokenizer.save_model(bpe_dir)

# Initialize WordPiece tokenizer
wordpiece_tokenizer = BertWordPieceTokenizer()

# Train tokenizer on training data
wordpiece_tokenizer.train_from_iterator(
    dataset["train"]["text"],
    vocab_size=30_512,
    min_frequency=2,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"]
)

# Save WordPiece tokenizer
wordpiece_tokenizer.save_model(wordpiece_dir)

# Save config.json manually
with open(os.path.join(wordpiece_dir, "config.json"), "w") as f:
    tokenizer_cfg = {
        "do_lower_case": True,
        "unk_token": "[UNK]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "mask_token": "[MASK]",
        "model_max_length": 1024,
    }
    json.dump(tokenizer_cfg, f)

# Benchmarking the tokenizers
# Random sample from dataset
sample_texts = random.sample(dataset["test"]["text"], 5)

# Tokenize with both tokenizers
for text in sample_texts:
    wordpiece_tokens = wordpiece_tokenizer.encode(text).tokens
    bpe_tokens = bpe_tokenizer.encode(text).tokens

    print(f"🔹 Original: {text}")
    print(f"🔸 WordPiece Tokens: {wordpiece_tokens}")
    print(f"🔹 Byte-Level BPE Tokens: {bpe_tokens}")
    print("=" * 80)

# Get vocab size
wordpiece_vocab_size = len(wordpiece_tokenizer.get_vocab())
bpe_vocab_size = len(bpe_tokenizer.get_vocab())

print(f"🔹 WordPiece Vocabulary Size: {wordpiece_vocab_size}")
print(f"🔸 Byte-Level BPE Vocabulary Size: {bpe_vocab_size}")

# Testing for OOV

def calculate_oov_rate(tokenizer, texts):
    total_tokens, unk_tokens = 0, 0
    for text in texts:
        encoded = tokenizer.encode(text)
        total_tokens += len(encoded.tokens)
        unk_tokens += encoded.tokens.count("[UNK]")
    
    return (unk_tokens / total_tokens) * 100

# Evaluate OOV Rate
oov_wordpiece = calculate_oov_rate(wordpiece_tokenizer, dataset["test"]["text"][:1000])
oov_bpe = calculate_oov_rate(bpe_tokenizer, dataset["test"]["text"][:1000])

print(f"🔹 WordPiece OOV Rate: {oov_wordpiece:.2f}%")
print(f"🔸 Byte-Level BPE OOV Rate: {oov_bpe:.2f}%")

