# FSMOD

![LICENSE](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.7-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-1.1.0-%237732a8)

The source code is based on  [https://github.com/facebookresearch/maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) and developed with Python 3.7 & PyTorch 1.1.0.

## Installation
Check INSTALL.md for installation instructions. Since maskrcnn-benchmark has been deprecated, please follow these instructions carefully (e.g. version of Python packages).

## Prepare FSMOD dataset
First, you need to download the FSMOD dataset [here](https://drive.google.com/file/d/14muqZUdbpnYQ_30ZpAP9KqrVVHSkJOhU/view?usp=sharing).
Then, put "datasets" into this repository.

## Training and Evaluation
1. Run the following for base training and novel training on Pascal VOC splits-1.

```bash
source activate python37
python tools/train_net.py --config-file "configs/pascal_voc/e2e_faster_rcnn_R_50_FPN_base.yaml"
```

2. Modify them if needed. If you have any question about these parameters (e.g. batchsize), please refer to [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) for quick solutions.
