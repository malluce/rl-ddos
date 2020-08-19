#!/bin/bash

DIR=$(dirname "${BASH_SOURCE[0]}")

export PATH="$PATH:$DIR/bin"
export PYTHONPATH="$PYTHONPATH:$DIR"
export TF_CPP_MIN_LOG_LEVEL=3

# Uncomment to enable tensorboard profiler
#export LD_LIBRARY_PATH="/usr/local/cuda/extras/CUPTI/lib64/:$LD_LIBRARY_PATH"

source "$DIR/env/bin/activate"
