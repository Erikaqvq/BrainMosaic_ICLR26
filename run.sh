#!/bin/bash
set -euo pipefail

# Train / eval with a single JSON config.
python main.py --config configs/train.example.json
