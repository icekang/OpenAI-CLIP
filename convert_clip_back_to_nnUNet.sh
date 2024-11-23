#!/bin/bash

source activate jepa

which python

python -u convert_clip_back_to_nnUNet.py --config configs/clip_shared_projector_shared_encoder_preivl_poststent_wd.yaml