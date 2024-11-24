#!/bin/bash

# Load Modules
source activate jepa

echo "My task ID: " $LLSUB_RANK
echo "Number of Tasks: " $LLSUB_SIZE

python scripts/correct_data_shape_and_direction_for_triplets.py $LLSUB_RANK $LLSUB_SIZE