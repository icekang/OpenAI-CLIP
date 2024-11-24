#!/bin/bash

# Load Modules
source activate jepa

echo "My task ID: " $LLSUB_RANK
echo "Number of Tasks: " $LLSUB_SIZE

python scripts/check_consistency.py $LLSUB_RANK $LLSUB_SIZE