import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256"
from transformers import (BertTokenizer, BertForMaskedLM, pipeline, AutoModelForMaskedLM, AutoTokenizer)
import torch
import math
import random
import numpy as np
import datasets
from datasets import load_dataset
import json
import itertools

# Data + filtering
slagelse = load_dataset('csv', data_files='/work/Ccp-OldNewsBERT_2024/data/validation data/Unseen_text.csv')
slagelse = slagelse.remove_columns(["Unnamed: 0", 'filnavn', 'dato', 'id', 'pwa'])

grundtvig = load_dataset('csv', data_files='/work/Ccp-OldNewsBERT_2024/data/validation data/Grundtvig_sample.csv')
grundtvig = grundtvig.remove_columns(['Unnamed: 0', 'id', 'title'])

# Picking random samples of unseen data
random.seed(42)
idxs = random.sample(range(len(slagelse['train'])), 1000)
batch = slagelse['train'].select(idxs)

# Appending sample to list
slagelse_sample = []
for ex in batch:
    slagelse_sample.append(ex)

slagelse_sample = list(itertools.chain(*[d.values() for d in slagelse_sample]))

# SAVE the data to a JSON file
with open('/work/Ccp-OldNewsBERT_2024/data/validation data/sample_data/slagelse_data.json', 'w', encoding='utf-8') as f:
    json.dump(slagelse_sample, f, ensure_ascii=False, indent=4)

# Grundtvig sampling
idxs = random.sample(range(len(grundtvig['train'])), 1000)
batch = grundtvig['train'].select(idxs)

# Appending sample to list
grundtvig_sample = []
for ex in batch:
    grundtvig_sample.append(ex)

grundtvig_sample = list(itertools.chain(*[d.values() for d in grundtvig_sample]))

# SAVE the data to a JSON file
with open('/work/Ccp-OldNewsBERT_2024/data/validation data/sample_data/grundtvig_sample.json', 'w', encoding='utf-8') as f:
    json.dump(grundtvig_sample, f, ensure_ascii=False, indent=4)