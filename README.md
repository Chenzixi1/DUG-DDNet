# DUG-DDNet
Released code for "DUG-DDNet:Deep Localization Uncertainty Guided Dual-branch Deep Fusion Network"


**Work in progress.**


## Installation
This repo is tested with Ubuntu 20.04, python==3.7, pytorch==1.4.0 and cuda==10.1

```bash
conda create -n dugddnet python=3.7

conda activate dugddnet
```

Install PyTorch and other dependencies:

```bash
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch

pip install -r requirements.txt
```

Build DCNv2 and the project
```bash
cd models/backbone/DCNv2

. make.sh

cd ../../..

python setup develop
```

## Data Preparation
Please download [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and organize the data as follows:

```
#ROOT		
  |training/
    |calib/
    |image_2/
    |depth_2/
    |label/
    |ImageSets/
  |testing/
    |calib/
    |image_2/
    |depth_2/
    |ImageSets/
```

Then modify the paths in config/paths_catalog.py according to your data path.

## Training & Evaluation

Training with one GPU. (TODO: The multi-GPU training will be further tested.)

```bash
CUDA_VISIBLE_DEVICES=0 python tools/plain_train_net.py --batch_size 8 --config runs/monoflex_depth.yaml --output output/exp
```

The model will be evaluated periodically (can be adjusted in the CONFIG) during training and you can also evaluate a checkpoint with

```bash
CUDA_VISIBLE_DEVICES=0 python tools/plain_train_net.py --config runs/monoflex_depth.yaml --ckpt YOUR_CKPT  --eval
```

You can also specify --vis when evaluation to visualize the predicted heatmap and 3D bounding boxes. The pretrained model for train/val split and logs are here.

**Note:** we observe an obvious variation of the performance for different runs and we are still investigating possible solutions to stablize the results, though it may inevitably due to the utilized uncertainties.

## Citation

If you find our work useful in your research, please consider citing:

```latex

```

## Acknowlegment

The code is heavily borrowed from [MonoFlex](https://github.com/zhangyp15/MonoFlex) and thanks for their contribution.
