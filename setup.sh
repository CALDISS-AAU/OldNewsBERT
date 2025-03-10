#!/usr/bin/env bash

# Denne fil skal køres, når man starter job på UCloud. Den installerer miljøet/pakkerne, som bruges i projektet

## installerer python pakker
pip install --upgrade pip # opgraderer pip

# OBS! Nedenstående linje skal rettes så stien passer med projektet
pip install -r /work/Ccp-OldNewsBERT_2024/requirements.txt