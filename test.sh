#!/usr/bin/env bash
DATA="car"
checkpoints="/opt/intern/users/xunwang/checkpoints"
CUDA_VISIBLE_DEVICES=7  python pool_test.py -data $DATA -r $checkpoints/bin/$DATA/512/1000_model.pkl
CUDA_VISIBLE_DEVICES=7  python pool_test.py -data $DATA -r $checkpoints/bin/$DATA/512/1050_model.pkl
CUDA_VISIBLE_DEVICES=7  python pool_test.py -data $DATA -r $checkpoints/bin/$DATA/512/1100_model.pkl
CUDA_VISIBLE_DEVICES=7  python pool_test.py -data $DATA -r $checkpoints/bin/$DATA/512/1150_model.pkl
CUDA_VISIBLE_DEVICES=7  python pool_test.py -data $DATA -r $checkpoints/bin/$DATA/512/1200_model.pkl

