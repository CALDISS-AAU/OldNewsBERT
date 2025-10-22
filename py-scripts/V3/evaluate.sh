#!/bin/bash
# install dependencies
pip install --upgrade pip # opgraderer pip
# OBS! Nedenstående linje skal rettes så stien passer med projektet
pip install -r "/work/Ccp-OldNewsBERT_2024/requirements.txt"
accelerate launch --num_processes=1 --mixed_precision=fp16 /work/Ccp-OldNewsBERT_2024/py-scripts/V3/04_evaluate_v2.py

