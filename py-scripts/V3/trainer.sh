#!/bin/bash
# install dependencies
pip install --upgrade pip # opgraderer pip
# OBS! Nedenstående linje skal rettes så stien passer med projektet
pip install -r /work/Ccp-OldNewsBERT_2024/requirements.txt

accelerate launch --multi_gpu --num_processes=4 --mixed_precision=fp16 /work/Ccp-OldNewsBERT_2024/py-scripts/V3/02_train.py