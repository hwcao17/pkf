# JPDAF Comparison
We implemented JPDAF based on the [StoneSoup library](https://stonesoup.readthedocs.io/en/stable/auto_tutorials/index.html). You can install it by `pip install stonesoup` We provide instructions to compare our PKF with JPDAF on simulated data.

## Data preparation
Download the [simulation data](https://drive.google.com/file/d/10IWuhw6gNsa33Dc5DMQ4FWDwWek4orCY/view?usp=drive_link) and extract it under `JPDAF_comparison/data`.

Or you can generate your own data using the script [generate_data.py](../JPDAF_comparison/generate_data.py)

```shell
cd JPDAF_comparison
python generate_data.py --n_obj 5 --noise_scale 0.75
```

## Evaluation

To run PKF with 5 objects
```shell
cd JPDAF_comparison
python run_mot_sim.py --n_obj 5
```
You may specify `--noise_scale` for experiments with 10 objects.

To run JPDAF with 5 objects
```shell
cd JPDAF_comparison
python run_mot_sim.py --n_obj 5 --jpdaf
```
You may specify `--noise_scale` for experiments with 10 objects.

To run PMHT with 5 objects
```shell
cd JPDAF_comparison
python run_mot_sim.py --n_obj 5 --pmht
```
You may specify `--noise_scale` for experiments with 10 objects.

Note that, we compute the data association weights with [StoneSoup library](https://stonesoup.readthedocs.io/en/stable/auto_tutorials/index.html) but the update step of JPDAF is implemented by ourselves. We also provide an update implementation with [StoneSoup library](https://stonesoup.readthedocs.io/en/stable/auto_tutorials/index.html) in [jpdaf_stonsoup.py](../JPDAF_comparison/jpdaf_stonesoup.py) which gives us similar tracking errors but much slower tracking rates. To run that script

```shell
python jpdaf_stonesoup.py --n_obj 5
```
You may specify `--noise_scale` for experiments with 10 objects.

