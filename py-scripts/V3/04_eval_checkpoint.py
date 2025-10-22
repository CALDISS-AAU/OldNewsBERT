import os
from os.path import join
import sys
import json
import torch
from torch.utils.data import DataLoader
import evaluate
import numpy as np
from pathlib import Path
import math
# import bertviz
import random
from datasets import *
import transformers
from transformers import (
    BertConfig, 
    BertForMaskedLM, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BertTokenizerFast,
    AutoModelForMaskedLM,
    AutoTokenizer,
    pipeline,
    set_seed
)
from huggingface_hub import(
    login,
    upload_folder,
    create_repo,
    HfApi
)

# DIRS AND PATHS
# Base project directory
project_dir = Path('/work/Ccp-OldNewsBERT_2024')

# Define core directories and paths
dataset_path = 'JohanHeinsen/ENO'
modelling_dir = project_dir / 'modelling' / 'V3'
output_dir = project_dir / 'output'

# Define subdirectories for modeling and logs
tokenizer_dir = project_dir / 'tokenizer'
wordpiece_dir = tokenizer_dir / 'wordpiece_tokenizer_v3'
model_dir = modelling_dir / 'models'
logs_dir = modelling_dir / 'V3_logs'

# Create subdirectories
tokenizer_dir.mkdir(exist_ok=True)
model_dir.mkdir(exist_ok=True)
logs_dir.mkdir(exist_ok=True)

# Load model + tokenizer
model_path = "/work/Ccp-OldNewsBERT_2024/modelling/V3/models/ON-BERT-v3/checkpoint-54000"
model = AutoModelForMaskedLM.from_pretrained(model_path)
tokenizer = BertTokenizerFast.from_pretrained('/work/Ccp-OldNewsBERT_2024/tokenizer/V3')

# Dataset
ENO = load_dataset(dataset_path)
ENO = ENO.filter(lambda x: x['pwa'] >= 0.9)
ENO = ENO['train'].train_test_split(test_size=0.2, seed=420)

set_seed(42)

# Tokenization
def encode_text(examples):
    '''Tokenizer with stride and truncation for dynamic padding'''
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        stride=256,
        return_overflowing_tokens=True,
        return_special_tokens_mask=True,
        padding=False
    )

# Tokenize datasets
ENO_tokenized = ENO.map(
    encode_text,
    batched=True,
    remove_columns=ENO['train'].column_names,
    desc="Tokenizing dataset"
)

# Split train/val/test
split_dataset = ENO_tokenized["train"].train_test_split(train_size=0.8, seed=42)
final_dataset = DatasetDict({
    "train": split_dataset["train"],
    "validation": split_dataset["test"],
    "test": ENO_tokenized["test"]
})

# Dataloaders
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, pad_to_multiple_of=8)


args = TrainingArguments(
    output_dir="/work/Ccp-OldNewsBERT_2024/output/eval_results",
    per_device_eval_batch_size=32,
    fp16=True,
    eval_strategy="no",
    report_to="none",
    seed=42
)

trainer = Trainer(
    model=model,
    args=args,
    eval_dataset=final_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer
)

eval_results = trainer.evaluate()
eval_results["perplexity"] = math.exp(eval_results["eval_loss"])
with open("/work/Ccp-OldNewsBERT_2024/output/eval_results/ON-BERT_v3_checkpoint-results-02-10-25.json", "w") as f:
    json.dump(eval_results, f, indent=4)
    print(eval_results)