#!/bin/bash
# install dependencies
pip install torch transformers numpy datasets sentencepiece
# Enable detailed error reporting for PyTorch distributed
# Run distributed training
torchrun --nproc_per_node=4 /work/Ccp-OldNewsBERT_2024/py-scripts/05_trainer.py