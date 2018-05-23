#!/usr/bin/env bash
DATA="cub"
checkpoints="/opt/intern/users/xunwang/checkpoints"
CUDA_VISIBLE_DEVICES=2  python pool_test.py -data $DATA -r $checkpoints/bin/$DATA/512/200_model.pkl
CUDA_VISIBLE_DEVICES=2  python pool_test.py -data $DATA -r $checkpoints/bin/$DATA/512/250_model.pkl
CUDA_VISIBLE_DEVICES=2  python pool_test.py -data $DATA -r $checkpoints/bin/$DATA/512/300_model.pkl
CUDA_VISIBLE_DEVICES=2  python pool_test.py -data $DATA -r $checkpoints/bin/$DATA/512/350_model.pkl
#CUDA_VISIBLE_DEVICES=7  python pool_test.py -data $DATA -r $checkpoints/bin/$DATA/512/1100_model.pkl
#CUDA_VISIBLE_DEVICES=7  python pool_test.py -data $DATA -r $checkpoints/bin/$DATA/512/1150_model.pkl
#CUDA_VISIBLE_DEVICES=7  python pool_test.py -data $DATA -r $checkpoints/bin/$DATA/512/1200_model.pkl

