import os
import huggingface_hub
import transformers
from transformers import BertTokenizer
from huggingface_hub import(
    login,
    upload_folder,
    create_repo,
    HfApi
)


# Loading model, tokenizer and dataset
model_path = '/work/Ccp-OldNewsBERT_2024/modelling/V3/models/ON-BERT-v3/checkpoint-120000'
# Loading custom tokenizer
tokenizer = BertTokenizer.from_pretrained('/work/Ccp-OldNewsBERT_2024/tokenizer/V3')
# Saving our pretrained tokenizer after training model
tokenizer.save_pretrained(model_path)

# Push to hub
# login
login()

# create repo
create_repo(repo_id='DA-BERT_Old_News_V3', repo_type='model', private=False)

# Get token and upload folder to repo
api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path=model_path,
    repo_id="SirMappel/DA-BERT_Old_News_V3",
    repo_type="model",
)