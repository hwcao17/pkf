#!/bin/bash

# default: run pkf
if [ $# -eq 0 ]; then
    python tools/run_dance.py -f exps/example/mot/yolox_dancetrack_val.py \
        -c pretrained/ocsort_dance_model.pth.tar \
        --detection_dir YOLOX_outputs/dancetrack/detections \
        --output_dir YOLOX_outputs/dancetrack -b 1 -d 1 --fp16 --fuse \
        --expn pkf --use_saved_dets \
        --ambig_thresh 0.9 --update_weight_thresh 0.3 --use_ocr --use_ocm
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
    python tools/run_dance.py --alg_name $1 -f exps/example/mot/yolox_dancetrack_val.py \
        -c pretrained/ocsort_dance_model.pth.tar \
        --detection_dir YOLOX_outputs/dancetrack/detections \
        --output_dir YOLOX_outputs/dancetrack -b 1 -d 1 --fp16 --fuse \
        --expn $1 --use_saved_dets 
    exit 1
fi
