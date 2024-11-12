#!/bin/bash

# default: run pkf
if [ $# -eq 0 ]; then
    python tools/run_mot.py -f exps/example/mot/yolox_x_mix_mot20_ch.py \
        -c pretrained/ocsort_x_mot20.pth.tar -b 1 -d 1 --fp16 --fuse --track_thresh 0.4 \
        --mot20 --test --expn pkf --ambig_thresh 0.95 --update_weight_thresh 0.25
    exit 1
fi

# run other algorithms
if [ $# -ne 0 ]; then
    # check if the algorithm name is valid
    if [ $1 != "ocsort" ] && [ $1 != "bytetrack" ]; then
        echo "Invalid algorithm name. Please provide one of the following: ocsort, bytetrack. 
              To run pkf, do not provide any arguments."
        exit 1
    fi
    python tools/run_mot.py --alg_name $1 -f exps/example/mot/yolox_x_mix_mot20_ch.py \
        -c pretrained/ocsort_x_mot20.pth.tar -b 1 -d 1 --fp16 --fuse --track_thresh 0.4 \
        --mot20 --test --expn $1 --ambig_thresh 0.95 --update_weight_thresh 0.25
    exit 1
fi
