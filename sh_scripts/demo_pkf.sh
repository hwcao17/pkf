python tools/demo_track_pkf.py --demo_type video \
    -f exps/example/mot/yolox_dancetrack_test.py -c pretrained/ocsort_dance_model.pth.tar \
    --path demo.mp4 --fp16 --fuse --save_result --out_path demo_out.mp4 \
    --ambig_thresh 0.9 --update_weight_thresh 0.3 --use_ocm --use_ocr