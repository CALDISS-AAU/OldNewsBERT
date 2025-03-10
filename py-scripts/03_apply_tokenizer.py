# Packages
import numpy as np
from pathlib import Path
import os
from transformers import AutoConfig
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizerFast
from transformers import PreTrainedTokenizerFast
from datasets import Dataset
from datasets import *
import sys
from os.path import join
import random
import pyarrow.csv as pv
import pyarrow.parquet as pq
from itertools import chain

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

# Loading data
dataset = load_from_disk('/work/Ccp-OldNewsBERT_2024/data/dataset')
dataset = load_from_disk(join(data_dir, 'dataset'))
train_dataset = dataset['train']
test_dataset = dataset['test']

tokenizer = BertTokenizerFast.from_pretrained(wordpiece_dir)

def tokenize_without_truncation(examples):
    """Tokenizes text but does NOT truncate, so we capture full token distributions."""
    tokens = tokenizer(
        examples["text"], truncation=False, padding=False, return_length=True
    )
    return {"length": tokens["length"]}  # Extract sequence length

# Apply tokenization to dataset
token_length_dataset = dataset.map(tokenize_without_truncation, batched=True)

# Convert lengths to a list
train_lengths = token_length_dataset["train"]["length"]
test_lengths = token_length_dataset["test"]["length"]

# Compute distribution stats
train_stats = {
    "min": np.min(train_lengths),
    "max": np.max(train_lengths),
    "mean": np.mean(train_lengths),
    "median": np.median(train_lengths),
    "percentile_95": np.percentile(train_lengths, 95),
    "percentile_99": np.percentile(train_lengths, 99),
}

test_stats = {
    "min": np.min(test_lengths),
    "max": np.max(test_lengths),
    "mean": np.mean(test_lengths),
    "median": np.median(test_lengths),
    "percentile_95": np.percentile(test_lengths, 95),
    "percentile_99": np.percentile(test_lengths, 99),
}

print("🔹 Train Token Length Stats:", train_stats)
print("🔹 Test Token Length Stats:", test_stats)

# Whether to truncate
truncate_longer_samples = False

# Max length
max_length = 512
stride = 256  # Allows overlap between chunks when truncating

# Tokenize function with & without truncation (including stride)
def encode_with_truncation(examples):
    """Tokenizes text with truncation and applies a sliding window."""
    return tokenizer(str(
        examples["text"]),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        stride=stride,  # Apply sliding window
        return_overflowing_tokens=True,  # Required to capture multiple chunks
        return_special_tokens_mask=True
    )

def encode_without_truncation(examples):
    """Tokenizes text without truncation."""
    return tokenizer(
        examples["text"],
        return_special_tokens_mask=True
    )

# Select encoding function based on truncate_longer_samples
encode = encode_with_truncation if truncate_longer_samples else encode_without_truncation

# Tokenize datasets
train_dataset = dataset["train"].map(encode, batched=True, remove_columns=["text"])
test_dataset = dataset["test"].map(encode, batched=True, remove_columns=["text"])

# Handle format
if truncate_longer_samples:
    # Ensure PyTorch compatibility
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
else:
    # Keep as lists for flexible processing
    test_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])
    train_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])

# Tokenization parameters
max_length = 512
stride = 256

def encode_text(examples):
    """Tokenizes text with padding, truncation, and stride."""
    return tokenizer(examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_special_tokens_mask=True
    )


# Tokenize datasets
train_dataset = dataset["train"].map(encode_text, batched=True, remove_columns=["text"])
test_dataset = dataset["test"].map(encode_text, batched=True, remove_columns=["text"])

# Set format for PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "special_tokens_mask"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "special_tokens_mask"])

########################################################################################################
# Tokenization parameters
max_length = 512
stride = 256


def encode_text(examples):
    """Tokenizes text with padding, truncation, and stride."""
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_special_tokens_mask=True
    )
    
    # Limit the number of tokens to match original batch size
    max_chunks = len(examples["text"])
    
    return {
        'input_ids': tokenized['input_ids'][:max_chunks],
        'attention_mask': tokenized['attention_mask'][:max_chunks],
        'special_tokens_mask': tokenized['special_tokens_mask'][:max_chunks]
    }

# Tokenize datasets
train_dataset = dataset["train"].map(
    encode_text, 
    batched=True, 
    remove_columns=["text"],
    batch_size=1000  # Explicitly set batch size to match your original batch
)
test_dataset = dataset["test"].map(
    encode_text, 
    batched=True, 
    remove_columns=["text"],
    batch_size=1000  # Explicitly set batch size to match your original batch
)

# Set format for PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "special_tokens_mask"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "special_tokens_mask"])


merged_dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})


# saving data
merged_dataset.save_to_disk('/work/Ccp-OldNewsBERT_2024/data/dataset_tokenized')