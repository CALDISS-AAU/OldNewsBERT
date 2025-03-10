#!/bin/bash
# install dependencies
pip install torch transformers numpy datasets sentencepiece
# Enable detailed error reporting for PyTorch distributed
# Run distributed training
torchrun /work/Ccp-OldNewsBERT_2024/py-scripts/05_trainer.py