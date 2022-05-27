export PYTHONPATH=/home/czx/Notebook/MonoFlex:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 python tools/plain_train_net.py --batch_size 8 --config runs/monoflex.yaml --output output/exp

CUDA_VISIBLE_DEVICES=0 python tools/plain_train_net.py --config runs/monoflex.yaml --ckpt output/pretrain/model_moderate_best_soft.pth  --eval --vis


# depth
CUDA_VISIBLE_DEVICES=1 python tools/plain_train_net.py --batch_size 4 --config runs/monoflex_depth.yaml --output output/base_depth/baseline3

CUDA_VISIBLE_DEVICES=0 python tools/plain_train_net.py --config runs/monoflex_depth.yaml --ckpt output/base_depth/baseline/model_moderate_best_soft.pth  --eval --vis

CUDA_VISIBLE_DEVICES=2 python tools/plain_train_net.py --batch_size 1 --config runs/monoflex_depth.yaml --output output/exp

CUDA_VISIBLE_DEVICES=0 python tools/plain_train_net.py --batch_size 4 --config runs/monoflex_depth.yaml --output output/base_depth/baseline4


CUDA_VISIBLE_DEVICES=2 python tools/plain_train_net.py --config runs/monoflex_depth.yaml --ckpt output/base_depth/baseline2_again/model_moderate_best_soft.pth  --eval --vis

CUDA_VISIBLE_DEVICES=1 python tools/plain_train_net.py --config runs/monoflex_depth_test.yaml --ckpt output/base_depth/newbaseline//model_moderate_best_soft.pth --eval

CUDA_VISIBLE_DEVICES=2 python tools/plain_train_net.py --batch_size 4 --config runs/monoflex_depth.yaml --output output/exp



CUDA_VISIBLE_DEVICES=0 python tools/plain_train_net.py --config runs/monoflex_depth_test.yaml --ckpt output/base_depth/newbaseline/dropout2_01/model_moderate_best_soft.pth --eval
CUDA_VISIBLE_DEVICES=1 python tools/plain_train_net.py --config runs/monoflex_depth_test.yaml --ckpt output/base_depth/newbaseline/dropout2_09/model_moderate_best_soft.pth --eval

