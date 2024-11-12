## Installation
PKF is built upon codebase of [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) and [OC-SORT](https://github.com/noahcao/OC_SORT). I tested the code with Python 3.8. 

### 1. Installing on the host machine
Step1. Install PKF
```shell
git clone https://github.com/hwcao17/deep_pkf_dev.git
cd pkf
pip install -r requirements.txt
python setup.py develop
cd trackers/pkf_tracker/data_association
python setup.py install
```

Step2. Install [pycocotools](https://github.com/cocodataset/cocoapi).

```shell
pip install cython; pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

Step3. Others
```shell
pip install cython_bbox pandas xmltodict
```


### 2. Download the pretrained models

Download the pretrained models from [Google Drive](https://drive.google.com/drive/folders/1LnhZVJlpufUnWuObZASIN1KwfhuvT_a8) and put them under `<pkf root>/pretrained`