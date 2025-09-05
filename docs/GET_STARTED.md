# Get Started 
We introduce the process of getting started on PKF. This instruction is adapted from ByteTrack especially for the training part. We provide some simple pieces here, for details please refer to the source code and *utils/args.py*.

## Data preparation

1. Download [MOT17](https://motchallenge.net/), [MOT20](https://motchallenge.net/), [DanceTrack](https://github.com/DanceTrack/DanceTrack) and put them under `/path/to/datasets` in the following structure and change [utils/data.py](../utils/data.py) line 12 to `yolox_datadir = /path/to/datasets`
    ```
    /path/to/datasets
    |——————mot
    |        └——————train
    |        └——————test
    └——————MOT20
    |        └——————train
    |        └——————test
    └——————dancetrack        
             └——————train
             └——————val
             └——————test
    ```

2. Turn the datasets to COCO format and mix different training data:

    ```python
    # replace "dance" with mot17/mot20 for others
    python3 tools/convert_dance_to_coco.py 
    ```

3. *[Optional]* If you want to train for MOT17/MOT20, follow the following to create mixed training set.

    ```python
    # build mixed training sets for MOT17 and MOT20 
    python3 tools/mix_data_{ablation/mot17/mot20}.py
    ```

## Evaluation

* **on DanceTrack Val set**
    ```shell
    python tools/run_dance.py -f exps/example/mot/yolox_dancetrack_val.py \
        -c pretrained/ocsort_dance_model.pth.tar \
        --detection_dir YOLOX_outputs/dancetrack/detections \
        --output_dir YOLOX_outputs/dancetrack -b 1 -d 1 --fp16 --fuse \
        --ambig_thresh 0.9 --update_weight_thresh 0.25 \
        --score_thresh 0.9 --skip_thresh 0.95 --use_ocr --coef 1.0 \
        --expn pkf --use_saved_dets
    ```
    or use the shell script [run_dance_val.sh](../sh_scripts/run_dance_val.sh). We follow the [TrackEval protocol](https://github.com/DanceTrack/DanceTrack/tree/main/TrackEval) for evaluation on the officially released validation set. This gives HOTA = 53.5. You may use flag `--save_detections` to save detected bounding boxes for future use.

* **on DanceTrack Test set**
    ```shell
    python tools/run_dance.py -f exps/example/mot/yolox_dancetrack_test.py \
        -c pretrained/ocsort_dance_model.pth.tar \
        --detection_dir YOLOX_outputs/dancetrack/detections \
        --output_dir YOLOX_outputs/dancetrack \
        -b 1 -d 1 --fp16 --fuse --ambig_thresh 0.9 --update_weight_thresh 0.25 \
        --score_thresh 0.8 --skip_thresh 0.9 --use_ocr --coef 1.0 \
        --expn pkf --test --use_saved_dets
    ```
    or use the shell script [run_dance_test.sh](../sh_scripts/run_dance_test.sh). Submit the outputs to [the DanceTrack evaluation site](https://competitions.codalab.org/competitions/35786). This gives HOTA = 55.4.

* **on MOT17 val**
    ```shell
    python tools/run_mot.py -f exps/example/mot/yolox_x_mot17_half.py \
        -c pretrained/ocsort_x_mot17.pth.tar -b 1 -d 1 --fp16 --fuse --expn pkf \
        --update_weight_thresh 0.25 --iou_thresh 0.3 --use_saved_dets
    ```
    or use the shell script [run_mot17_val.sh](../sh_scripts/run_mot17_val.sh). We follow the [TrackEval protocol](https://github.com/DanceTrack/DanceTrack/tree/main/TrackEval) for evaluation on the self-splitted validation set. This gives you HOTA = 76.9.

* **on MOT17/MOT20 Test set**
    ```shell
    # MOT17
    python tools/run_mot.py -f exps/example/mot/yolox_x_mix_det.py \
        -c pretrained/ocsort_x_mot17.pth.tar -b 1 -d 1 --fp16 --fuse --test \
        --expn pkf --use_saved_dets --update_weight_thresh 0.25

    # MOT20
    python tools/run_mot.py -f exps/example/mot/yolox_x_mix_mot20_ch.py \
        -c pretrained/ocsort_x_mot20.pth.tar -b 1 -d 1 --fp16 --fuse --track_thresh 0.4 \
        --mot20 --test --expn pkf --ambig_thresh 0.95 --update_weight_thresh 0.25
    ```
    or use shell scripts [run_mot17_test.sh](../sh_scripts/run_mot17_test.sh) and [run_mot20_test.sh](../sh_scripts/run_mot20_test.sh). Submit the zipped output files to [MOTChallenge](https://motchallenge.net/) system. Following [the adaptive detection thresholds](https://github.com/ifzhang/ByteTrack/blob/d742a3321c14a7412f024f2218142c7441c1b699/yolox/evaluators/mot_evaluator.py#L139) by ByteTrack can further boost the performance. After interpolation (see below), this gives you HOTA = 63.3 on MOT17 and HOTA = 62.3 on MOT20.

* **on MOT17/MOT20 Train set**
    ```shell
    # MOT17
    python tools/run_mot.py -f exps/example/mot/yolox_x_mot17_train.py \
        -c pretrained/ocsort_x_mot17.pth.tar -b 1 -d 1 --fp16 --fuse --expn pkf --train \
        --ambig_thresh 0.9 --update_weight_thresh 0.25 --use_saved_dets

    # MOT20
    python tools/run_mot.py -f exps/example/mot/yolox_x_mot20_train.py \
        -c pretrained/ocsort_x_mot20.pth.tar -b 1 -d 1 --fp16 --fuse \
        --track_thresh 0.4 --mot20 --expn pkf --train --use_saved_dets \
        --ambig_thresh 0.95 --update_weight_thresh 0.25
    ```
    or use shell scripts [run_mot17_test.sh](../sh_scripts/run_mot17_test.sh) and [run_mot20_test.sh](../sh_scripts/run_mot20_test.sh). We follow the [TrackEval protocol](https://github.com/DanceTrack/DanceTrack/tree/main/TrackEval) for evaluation on the train set. This gives you HOTA = 74.7 on MOT17 and HOTA = 77.4 on MOT20.

## [Optional] Interpolation
PKF is designed for online tracking, but offline interpolation has been demonstrated efficient for many cases. To use the linear interpolation over existing tracking results:
```shell
    # optional offline post-processing
    python3 tools/interpolation.py $result_path $save_path
```
Furthermore, we provide a piece of attempt of using Gaussian Process Regression in interpolating trajectories, which work upon existing linear interpolation results:
```shell
    python3 tools/gp_interpolation.py $raw_results_path $linear_interp_path $save_path
```
*Note: for the results in our paper on MOT17/MOT20 private settings, we use linear interpolation by default.*