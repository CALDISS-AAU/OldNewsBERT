# Packages
from pathlib import Path
import os
from transformers import AutoConfig
from tokenizers import ByteLevelBPETokenizer
from datasets import Dataset
import pandas as pd
import json
import sys
from os.path import join
import pyarrow.csv as pv
import pyarrow.parquet as pq
import dask.dataframe as dd

# DIRS AND PATHS
project_dir = join('/work', 'Ccp-OldNewsBERT_2024')
data_dir = join(project_dir, 'data')
modelling_dir = join(project_dir, 'modelling')
os.makedirs(modelling_dir, exist_ok=True)
modules_dir = join(project_dir, 'modules')
sys.path.append(modules_dir)

logs_dir = join(modelling_dir, 'logs')
output_dir = join(project_dir, 'output')
model_dir = join(modelling_dir, 'models')

# Converting .csv file to parquet because reasons
table = pv.read_csv("/work/Ccp-OldNewsBERT_2024/data/ONBert_data.csv", parse_options=pv.ParseOptions(delimiter=";"))
pq.write_table(table,'dataset.parquet' , compression="snappy")

# reading parquet data
df = dd.read_parquet("/work/Ccp-OldNewsBERT_2024/data/dataset.parquet")

#keep dataframe in memory to reduce compute
df = df.persist()

# removing this column for better compute also because it is irrelevant to task
df = df.drop(columns=['lignende_tekster', 'kontekst'])

def txt_len(row):
    return len(row['text'])  #compute the length of 'text'

# Apply function row-wise using apply (with axis=1 for rows)
df['text_length'] = df.apply(txt_len, axis=1, meta=('text_length', 'int'))

#removing rows where text_length is less than 10 char long
df = df.loc[df.text_length >= 30] #max == 34653
df = df.drop(columns='text_length')

# strip lines function
def remove_lines(text):
    if isinstance(text, str):
        return text.strip()
    return text

df['text'] = df['text'].map(remove_lines, meta=('text', 'str'))

# Convert Dask DataFrame to Pandas
df = df.compute()

# Reset index
df = df.reset_index(drop=True)

# Convert to dictionary format for Hugging Face Datasets
data_dict = df.to_dict(orient='list')

# Convert to Dataset
dataset = Dataset.from_dict(data_dict)

# Flatten (if nested features exist)
dataset = dataset.flatten()

# Split into train/test
dataset = dataset.train_test_split(test_size=0.1, seed=666)

# saving data
dataset.save_to_disk('/work/Ccp-OldNewsBERT_2024/data/dataset')
