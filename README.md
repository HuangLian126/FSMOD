# FSMOD

![LICENSE](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.7-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-1.1.0-%237732a8)

The source code is based on  [https://github.com/facebookresearch/maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) and developed with Python 3.7 & PyTorch 1.1.0.

## Installation
1. cd FSMOD/apex
2. python setup.py install --cuda_ext --cpp_ext
3. cd FSMOD
4. python setup.py build develop

## Prepare FSMOD dataset
First, you need to download the FSMOD dataset [here](https://drive.google.com/file/d/1B56_tgmj6P-aOU9t-vT4x92YuZrnFz2I/view?usp=share_link).
Then, put "datasets" into this repository.

## Training and Evaluation
1. Run the following for base-training and novel-fine-tuning.

```bash
python tools/train_net.py --config-file "configs/pascal_voc/e2e_faster_rcnn_R_50_FPN_base.yaml"
python classHead_FCiniti.py
python tools/train_net.py --config-file "configs/pascal_voc/e2e_faster_rcnn_R_50_FPN_novel.yaml"
```

2. Modify them if needed. If you have any question about these parameters (e.g. batchsize), please refer to [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) for quick solutions.
