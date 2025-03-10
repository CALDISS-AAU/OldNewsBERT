# Packages
import numpy as np
import sys
import os
from os.path import join
from datasets import Dataset
from datasets import *
import accelerate
import transformers
from transformers import (
    BertConfig, 
    BertForMaskedLM, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BertTokenizerFast
)
import torch

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

# Loading our tokenized dataset
dataset = load_from_disk(join(data_dir, 'dataset_tokenized'))

# splitting dataset again to create a validation set
split_dataset = dataset["train"].train_test_split(train_size=0.8, seed=42)

# Setting up the data format from before
final_dataset = DatasetDict({
    "train": split_dataset["train"],
    "validation": split_dataset["test"],  # The test portion from the split becomes validation
    "test": dataset["test"]  # Keep the original test set
})

# Loading custom tokenizer
tokenizer = BertTokenizerFast.from_pretrained(wordpiece_dir)

# 30,522 vocab is BERT's default vocab size, feel free to tweak
vocab_size = 30_522
# maximum sequence length, lowering will result to faster training (when increasing batch size)
max_length = 512

# initialize the model with the config
model_config = BertConfig(vocab_size=vocab_size, max_position_embeddings=max_length)
model = BertForMaskedLM(config=model_config)

### Set device
device = "cuda"
model.to(device)

# initialize the data collator, randomly masking 20% (default is 15%) of the tokens for the Masked Language
# Modeling (MLM) task
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.2
)


# Enhanced Training arguments
training_args = TrainingArguments(
    output_dir=modelling_dir,
    eval_strategy="steps", # Evaluate each 'logging_stops' steps
    overwrite_output_dir=True,
    num_train_epochs=15,
    per_device_train_batch_size=16, # training batch size
    gradient_accumulation_steps=4, # Accumulating the gradients before updating the weights
    per_device_eval_batch_size=64, # Evaluating batch size
    logging_steps=500,
    learning_rate=5e-5, # defining learning rate
    save_steps=1000,
    save_total_limit=5,           # Keep only the 5 best checkpoints to save disk space
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",  # Specify which metric to use for best model
    greater_is_better=False,      # For loss, lower is better
    fp16=torch.cuda.is_available(),
    warmup_steps=10000,           # Gradual warmup to prevent training instability
    warmup_ratio=0.06,
    weight_decay=0.01,            # L2 regularization to prevent overfitting
    lr_scheduler_type="cosine",   # Cosine learning rate decay
    dataloader_num_workers=4,     # Speed up data loading with multiple workers
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
    eval_dataset=final_dataset['validation'],
)

# train the model
trainer.train()