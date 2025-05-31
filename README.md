# SemInv

## Enrionmet Setup

### Python Environment

conda create -n seminv python=3.8

pip install -r requirements.txt

### Dataset Prepare

visit https://pages.nist.gov/trojai/, find round 6-8, download correponding datasets.

## Quick Start

### Trigger Inversion

`python src/main.py`

### Condition Test

`python src/condition.py`

### Evaluate

For accuracy, `python src/acc.py`

For trigger accuracy, `python src/trigger_acc.py`
