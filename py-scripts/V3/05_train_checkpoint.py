print("Running from:", __file__)
# Packages
import numpy as np
import sys
import os
from pathlib import Path
from os.path import join
import math
from datasets import Dataset
from datasets import *
import accelerate
import transformers
from transformers import (
    BertConfig, 
    BertForMaskedLM,
    AutoModelForMaskedLM, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BertTokenizerFast,
    set_seed
)
import torch
import torch.distributed as dist

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# DIRS AND PATHS
# Base project directory
project_dir = Path('/work/Ccp-OldNewsBERT_2024')

# Define core directories and paths
dataset_path = 'JohanHeinsen/ENO'
modelling_dir = project_dir / 'modelling' / 'V3'
output_dir = project_dir / 'output'

# Define subdirectories for modeling and logs
tokenizer_dir = modelling_dir / 'tokenizers'
wordpiece_dir = tokenizer_dir / 'wordpiece_tokenizer_v3'
model_dir = modelling_dir / 'models'
logs_dir = modelling_dir / 'V3_logs'

# Create subdirectories
tokenizer_dir.mkdir(exist_ok=True)
model_dir.mkdir(exist_ok=True)
logs_dir.mkdir(exist_ok=True)

# Load Dataset
ENO = load_dataset(dataset_path)

# Filter dataset based on pwa values
ENO = ENO.filter(lambda accuracy : accuracy['pwa'] >= 0.9)

# Train_test split
ENO = ENO['train'].train_test_split(test_size=0.2, seed=420)

# Loading custom tokenizer + model
tokenizer = BertTokenizerFast.from_pretrained('/work/Ccp-OldNewsBERT_2024/tokenizer/V3')
# Load model
model_path = ('/work/Ccp-OldNewsBERT_2024/modelling/V3/models/ON-BERT-v3/checkpoint-57020')
model = AutoModelForMaskedLM.from_pretrained(model_path)
set_seed(42)
# Tokenize function
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
tokenized_datasets = ENO.map(
    encode_text,
    batched=True,
    remove_columns=ENO['train'].column_names,
    desc="Tokenizing dataset"
)

# splitting for validation set
split_dataset = tokenized_datasets["train"].train_test_split(
    train_size=0.8, 
    seed=42
    )

final_dataset = DatasetDict({
    "train": split_dataset["train"],
    "validation": split_dataset["test"],
    "test": tokenized_datasets["test"]
})

# Set PyTorch format
final_dataset.set_format(
    type="torch", 
    columns=["input_ids", "attention_mask", "special_tokens_mask"]
)

# Modeling (MLM) task
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=True, 
    mlm_probability=0.15,
    pad_to_multiple_of=8
)

# Training arguments
training_args = TrainingArguments(
    output_dir='/work/Ccp-OldNewsBERT_2024/modelling/V3/models/ON-BERT-v3',
    eval_strategy="steps",
    overwrite_output_dir=False,
    # num_train_epochs=5,
    per_device_train_batch_size=16, # training batch size
    gradient_accumulation_steps=4, # Accumulating the gradients before updating the weights
    per_device_eval_batch_size=32, # Evaluating batch size
    logging_steps=500,
    learning_rate=5e-5, # defining learning rate
    save_steps=3000,
    max_steps=98640,
    save_total_limit=5,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=torch.cuda.is_available(),
    warmup_ratio=0.06,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    save_on_each_node=False,
    ddp_find_unused_parameters=False,
    optim="adamw_torch",
)

# initialize the trainer and pass everything to it
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=final_dataset['train'],
    eval_dataset=final_dataset['test'],
)

# train the model
trainer.train(resume_from_checkpoint='/work/Ccp-OldNewsBERT_2024/modelling/V3/models/ON-BERT-v3/checkpoint-57020')