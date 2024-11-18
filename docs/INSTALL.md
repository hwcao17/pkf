## Installation
PKF is built upon codebase of [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) and [OC-SORT](https://github.com/noahcao/OC_SORT). I tested the code with Python 3.8. 

### 1. Installing on the host machine
Step1. Install dependencies
```shell
git clone https://github.com/hwcao17/pkf.git
cd pkf
pip install -r requirements.txt
python setup.py develop
```

Install [Eigen3](https://eigen.tuxfamily.org/dox/GettingStarted.html), you may use
```shell
sudo apt install libeigen3-dev
```

Install [pybind11](https://pybind11.readthedocs.io/en/latest/installing.html#include-with-pypi), you may use
```shell
pip install pybind11
```

Step2. Install the probabilistic data association
```shell
cd trackers/pkf_tracker/data_association
python setup.py install
```
If you have problem with `pybind11` at this step, you may consider installing `pybind11` using
```shell
pip install "pybind11[global]"
```

Step3. Install [pycocotools](https://github.com/cocodataset/cocoapi).

```shell
pip install cython; pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

Step4. Others
```shell
pip install cython_bbox pandas xmltodict
```


### 2. Download the pretrained models

Download the pretrained models from [Google Drive](https://drive.google.com/drive/folders/1LnhZVJlpufUnWuObZASIN1KwfhuvT_a8) and put them under `<pkf root>/pretrained`