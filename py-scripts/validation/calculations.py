import json
import pandas as pd 
import numpy
import os
from pathlib import Path

file = '/work/Ccp-OldNewsBERT_2024/output/eval_results/xlm-roberta-base-slagelse-ppl_results.json'
output_path = '/work/Ccp-OldNewsBERT_2024/output/ppl_stats_results'


with open(file) as f:
    data = json.load(f)

print(len(data["results"]))  # should print 1000 if complete

finished = set(data["results"].keys())



def calculate_stats(file, output_path, model_name, sample_name):
    output_path = f"{output_path}/{model_name}_{sample_name}_stats.csv"
    df = pd.read_json(file)
    df['ppl_value'] = df['results'].apply(lambda x: x['ppl'])
    base_stats = df.describe()
    base_stats.loc["median"] = df["ppl_value"].median()
    base_stats.to_csv(output_path)
    return base_stats

calculate_stats(file, output_path, "XLM-Roberta-base", "Slagelse")