#!/bin/bash

# default: run pkf
if [ $# -eq 0 ]; then
    python tools/run_dance.py -f exps/example/mot/yolox_dancetrack_test.py \
        -c pretrained/ocsort_dance_model.pth.tar \
        --detection_dir YOLOX_outputs/dancetrack/detections \
        --output_dir YOLOX_outputs/dancetrack \
        -b 1 -d 1 --fp16 --fuse --ambig_thresh 0.9 --update_weight_thresh 0.3 \
        --expn pkf --test --use_saved_dets --use_ocm --use_ocr
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
    python tools/run_dance.py -alg_name $1 -f exps/example/mot/yolox_dancetrack_test.py \
        -c pretrained/ocsort_dance_model.pth.tar \
        --detection_dir YOLOX_outputs/dancetrack/detections \
        --output_dir YOLOX_outputs/dancetrack \
        -b 1 -d 1 --fp16 --fuse --expn $1 --test --use_saved_dets
    exit 1
fi
