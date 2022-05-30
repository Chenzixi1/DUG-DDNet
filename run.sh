export PYTHONPATH=/home/czx/Notebook/MonoFlex:$PYTHONPATH

# trian 
CUDA_VISIBLE_DEVICES=0 python tools/plain_train_net.py --batch_size 4 --config runs/monoflex_depth.yaml --output output/exp

CUDA_VISIBLE_DEVICES=0 python tools/plain_train_net.py --batch_size 4 --config runs/monoflex_depth0.yaml --output output/exp

# test
CUDA_VISIBLE_DEVICES=0 python tools/plain_train_net.py --config runs/monoflex_depth_test.yaml --ckpt output/pretrain/model_moderate_best_soft.pth  --eval --vis

CUDA_VISIBLE_DEVICES=0 python tools/plain_train_net.py --config runs/monoflex_depth0_test.yaml --ckpt output/pretrain/model_moderate_best_soft.pth  --eval --vis
