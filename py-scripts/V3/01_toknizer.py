# Packages
from pathlib import Path
import os
from transformers import AutoConfig
from tokenizers import BertWordPieceTokenizer
from datasets import *
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
import json
import sys
from os.path import join
import random

# DIRS AND PATHS
# Base project directory
project_dir = Path('/work/Ccp-OldNewsBERT_2024')

# Define core directories and paths
dataset_path = 'JohanHeinsen/ENO'
modelling_dir = project_dir / 'modelling' / 'V3'
output_dir = project_dir / 'output'

# Define subdirectories for modeling and logs
tokenizer_dir = project_dir / 'tokenizer'
logs_dir = modelling_dir / 'V3_logs'

# Create subdirectories
tokenizer_dir.mkdir(exist_ok=True)
logs_dir.mkdir(exist_ok=True)

# Load Dataset
ENO = load_dataset(dataset_path)

# Filter dataset based on pwa values
ENO = ENO.filter(lambda accuracy : accuracy['pwa'] >= 0.9)

# Shuffeling dataset
ENO = ENO.shuffle(seed=666)

# Train_test split
ENO = ENO['train'].train_test_split(test_size=0.2, seed=420)

# Initialize WordPiece tokenizer
wordpiece_tokenizer_v3 = BertWordPieceTokenizer()

# Train tokenizer on training data
wordpiece_tokenizer_v3.train_from_iterator(
    ENO["train"]["text"],
    vocab_size=30_512,
    min_frequency=2,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"]
)

# Save WordPiece tokenizer
wordpiece_tokenizer_v3.save_model('/work/Ccp-OldNewsBERT_2024/tokenizer/V3')

# Save config.json
with open(os.path.join('/work/Ccp-OldNewsBERT_2024/tokenizer/V3', "config.json"), "w") as f:
    tokenizer_cfg = {
        "do_lower_case": True,
        "unk_token": "[UNK]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "mask_token": "[MASK]",
        "model_max_length": 512,
    }
    json.dump(tokenizer_cfg, f)

