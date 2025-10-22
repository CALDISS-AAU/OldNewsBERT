import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256"
from transformers import (BertTokenizer, BertForMaskedLM, pipeline, AutoModelForMaskedLM, AutoTokenizer, RobertaForMaskedLM)
import torch
import math
import random
import numpy as np
import datasets
from datasets import load_dataset
import json
import itertools

# Data + filtering
# Slagelse
with open('/work/Ccp-OldNewsBERT_2024/data/validation data/sample_data/slagelse_sample.json', 'r', encoding='utf-8') as f:
    slagelse_sample = json.load(f)

# Grundvig
with open('/work/Ccp-OldNewsBERT_2024/data/validation data/sample_data/grundtvig_sample.json', 'r', encoding='utf-8') as f:
    grundtvig_sample = json.load(f)


# Loading out model
model_name = 'FacebookAI/xlm-roberta-base'
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Setting up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def score(model, tokenizer, sentence):
    """
    Function for calculating Pseudo Perplexity (PPL).
    Masks each token in the sequence using the loaded model
    (ignoring [CLS] and [SEP] tokens).
    Safe for GPU execution with limited memory.
    Returns a dict with the calculated PPL.
    """

    # --- Device setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # --- Tokenization ---
    tensor_input = tokenizer.encode(
        sentence,
        return_tensors="pt",
        max_length=512,
        truncation=True
    ).to(device)

    # print("Model on:", next(model.parameters()).device)
    # print("Tensor on:", tensor_input.device)

    seq_len = tensor_input.size(-1)
    mask = torch.ones(seq_len - 1).diag(1)[:-2].to(device)
    # print("Mask on:", mask.device)

    num_masks = seq_len - 2
    batch_size = 4  # smaller batch for safe GPU use
    partial_losses = []

    # --- Loop through masked batches ---
    for start in range(0, num_masks, batch_size):
        end = min(start + batch_size, num_masks)
        batch_mask = mask[start:end]
        batch_input = tensor_input.repeat(end - start, 1)

        masked_input_batch = batch_input.masked_fill(batch_mask == 1, tokenizer.mask_token_id)
        label_batch = batch_input.masked_fill(masked_input_batch != tokenizer.mask_token_id, -100)

        with torch.inference_mode():
            outputs = model(masked_input_batch, labels=label_batch)
            loss = outputs.loss

        partial_losses.append(loss.item())

        # Free memory between batches
        del masked_input_batch, label_batch, outputs, loss
        torch.cuda.empty_cache()

    # --- Compute mean loss and PPL ---
    mean_loss = sum(partial_losses) / len(partial_losses)
    ppl = np.exp(mean_loss)

    return {"ppl": ppl}

results = {}
for text in slagelse_sample:
    ppl = score(model, tokenizer, text)
    results[text] = ppl
output = {"model": "xlm-roberta-base", "results": results}


with open(f"/work/Ccp-OldNewsBERT_2024/output/eval_results/xlm-roberta-base-slagelse-ppl_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4, ensure_ascii=False)
    f.flush()
    os.fsync(f.fileno())