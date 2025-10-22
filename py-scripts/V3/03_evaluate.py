import os
from os.path import join
import sys
import torch
from torch.utils.data import DataLoader
import evaluate
import numpy as np
from pathlib import Path
import math
import bertviz
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
    pipeline
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
model_path = "/work/Ccp-OldNewsBERT_2024/modelling/V3/models/ON-BERT-v3/checkpoint-57020"
model = AutoModelForMaskedLM.from_pretrained(model_path)
tokenizer = BertTokenizerFast.from_pretrained('/work/Ccp-OldNewsBERT_2024/tokenizer/V3')

# Dataset
ENO = load_dataset(dataset_path)
ENO = ENO.filter(lambda x: x['pwa'] >= 0.9).shuffle(seed=666)
ENO = ENO['train'].train_test_split(test_size=0.2, seed=420)

# Tokenization
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_special_tokens_mask=True
    )

ENO_tokenized = ENO.map(
    tokenize_function,
    batched=True,
    remove_columns=ENO['train'].column_names)

# Split train/val/test
split_dataset = ENO_tokenized["train"].train_test_split(train_size=0.8, seed=42)
final_dataset = DatasetDict({
    "train": split_dataset["train"],
    "validation": split_dataset["test"],
    "test": ENO_tokenized["test"]
})

# final_dataset.set_format(
#     type="torch", 
#     columns=["input_ids", "attention_mask", "special_tokens_mask"]
# )

# Dataloaders
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)
test_dataloader = DataLoader(final_dataset["test"], batch_size=16, collate_fn=data_collator)
valid_dataloader = DataLoader(final_dataset["validation"], batch_size=16, collate_fn=data_collator)

# Model eval
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

total_loss = 0
num_batches = 0

with torch.no_grad():
    for i, batch in enumerate(test_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss.item()
        total_loss += loss
        num_batches += 1
        print(f"Batch {i+1}/{len(test_dataloader)} Loss: {loss:.4f}")

avg_loss = total_loss / num_batches
print(f"Average test loss: {avg_loss:.4f}, Perplexity: {math.exp(avg_loss):.2f}")



# inference check
# Number of random rows to print
num_rows = 5

# Get random indices from the dataset
random_indices = random.sample(range(len(valid_data)), num_rows)

# Print the random rows
for index in random_indices:
    print(valid_data[index]['text'])
    print( "=" * 50)

text =['Naar Banken har gjort Udlaan, imod haandfaaet Pant, og den udlaante Sum ikke til den bestemte Tid betales tilbageeller Laanet ikke, hvis Directeurerne det tienligt eragte, fornyes eller forlænges, da skal Banken være berettiget tiluden foregaaende Dom eller Jndførsel, at lade, ved dens egne Betiente, og paa hvad Sted, den finder beleiligt, holde [MASK] over det, den saaledes til haandfaaet Pant forskrevet eller overleveret er, og ikke til rette Tid er indfriet.',
'Det Borgermands Barn, som stod i sidste Avis, kan antages i lille Torvegaden Nr. 89 i Stuen hos en [MASK]. Det Borgermands Barn, som forlangte Kost, Logis og at lære Skrædersom, kan bekomme det i Pilestræde Nr. 75 første Sal, hvor hun tillige kan lære alle andre FruentimmerNetheder for billig Betaling. Sengklæder forlanges til Leye, Adressen er fra Contoiret. Omtrent for to Maaneder siden var adskillige gange en Mand hos mig i No. 120 paa Uldfeldsplads øverste Bagsal, forat bestille Logis for en Person; men som denne Mands Boepæl er mig ubekiendt, udbeder jeg mig den Godhed af ham, endnu en gang at tale med mig.',
'Løverdagen den 12te Junii førstkommende, om Eftermiddagen imellem 3 og 5 Slet, skal ved offentlig Auction Anden Gang opbydes og til de Høystbydende bortsælges afg. Snedker Baltzers og efterlevende Hustrues Fælleds Stervboes tilhørende og i Tornebuskegaden beliggende [MASK].',
'Almindeligen fortælles en Anecdote om vor ny Minister for de udenlandske Anliggender, der [MASK] ham meget Ære.',
'Løverdagen den 12te Augusti Klokken 3 slet Eftermiddag, foretages Auction i Sr. Johan Qvists Leyegaard i nedre Voldgaden over afdøde Naalemager Johan Mechs efterladte Løsøre, [MASK] af nogle faa Meubler og Værktøy.'
]
mask_filler = pipeline("fill-mask", model, tokenizer=tokenizer)
mask_filler(text, top_k=2)

for ex in text:
    print(f"Input: {ex}")
    print(mask_filler(ex))
    print("=" * 50)


# To typer: Annonce & nyhedsstoffet.
# Skævvridning i learning ved annoncestoffet grundet den type skrift der forkommer.
# CHC har lavet classifier. Evt bruge den til at hjælpe med skalering
# ift. validering: værdisætning af en domænespecifik model. sammenlign med øvrige modeller til historisk materiale.
# MiMe-MeMo & multilinguale modeller

# Push to hub
# login
login()

# create repo
create_repo(repo_id='DA-BERT_Old_News_V1', repo_type='model', private=False)

# Get token and upload folder to repo
api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path=model_path,
    repo_id="SirMappel/DA-BERT_Old_News_V1",
    repo_type="model",
)

import pandas as pd
import matplotlib.pyplot as plt

log_history = '/work/Ccp-OldNewsBERT_2024/modelling/checkpoint-158840/trainer_state.json'
df = pd.DataFrame(log_history)
df[['step', 'loss']].dropna().plot(x='step', y='loss', title='Training Loss Curve')
plt.show()


# ------------------------------------- Memo check ----------------------------------------

model = AutoModelForMaskedLM.from_pretrained("MiMe-MeMo/MeMo-BERT-03")
tokenizer = AutoTokenizer.from_pretrained("MiMe-MeMo/MeMo-BERT-03", use_fast=True)

# Loading data
dataset = load_from_disk(join(data_dir, 'V2_dataset_dict'))
test_data = dataset['test']
valid_data = dataset['valid']

# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True, 
        padding="max_length", 
        max_length=512,  # Adjust max length as needed
        # return_tensors="pt"  # Ensures PyTorch tensors are returned
    )

tokenized_test = test_data.map(tokenize_function, batched=True)
tokenized_valid = valid_data.map(tokenize_function, batched=True)

tokenized_test = tokenized_test.remove_columns(["text", "id"])
tokenized_valid = tokenized_valid.remove_columns(['text', 'id'])

# Computing loss
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)

test_dataloader = DataLoader(tokenized_test, batch_size=16, collate_fn=data_collator)
valid_dataloader = DataLoader(tokenized_valid, batch_size=16, collate_fn=data_collator)
model.eval()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

total_loss = 0
num_batches = 0

with torch.no_grad():
    for i, batch in enumerate(test_dataloader):
        print(f"Processing batch {i+1}/{len(test_dataloader)}")
        batch = {k: v.to(device) for k, v in batch.items()}  # Move batch to CUDA
        outputs = model(**batch)
        loss = outputs.loss.item()

        total_loss += loss
        num_batches += 1                 

        print(f"Batch {i+1} Loss: {loss:.4f}")

avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")
print(f"Test Loss: {avg_loss:.4f}") # equals to 4.5983
perplexity = math.exp(avg_loss)
print(perplexity) # 99.31534399999609


# Push to hub
# login
login()

# create repo
create_repo(repo_id='DA-BERT_Old_News_V2', repo_type='model', private=False)

# Get token and upload folder to repo
api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path=model_path,
    repo_id="SirMappel/DA-BERT_Old_News_V2",
    repo_type="model",
)
