#!/usr/bin/env bash
DATA="cub"
checkpoints="/opt/intern/users/xunwang/checkpoints"

#CUDA_VISIBLE_DEVICES=0 python test.py -data $DATA -r $checkpoints/bin/$DATA/512-theta0/200_model.pkl
#CUDA_VISIBLE_DEVICES=0 python test.py -data $DATA -r $checkpoints/bin/$DATA/512-theta0.1/200_model.pkl
#CUDA_VISIBLE_DEVICES=4 python test.py -data $DATA -r $checkpoints/bin/$DATA/64-ABE/200_model.pkl
##CUDA_VISIBLE_DEVICES=7 python test.py -data $DATA -r $checkpoints/bin/$DATA/512-theta0/300_model.pkl
#CUDA_VISIBLE_DEVICES=6 python test.py -data $DATA -r $checkpoints/bin/$DATA/128-theta1/300_model.pkl
##CUDA_VISIBLE_DEVICES=6 python test.py -data $DATA -r $checkpoints/bin/$DATA/128-theta1/350_model.pkl
#CUDA_VISIBLE_DEVICES=6 python test.py -data $DATA -r $checkpoints/bin/$DATA/512-theta0/100_model.pkl
#CUDA_VISIBLE_DEVICES=6 python test.py -data $DATA -r $checkpoints/bin/$DATA/512-theta0/200_model.pkl

#CUDA_VISIBLE_DEVICES=6 python test.py -data $DATA -r $checkpoints/bin/$DATA/512-theta0/300_model.pkl
CUDA_VISIBLE_DEVICES=6 python test.py -data $DATA -r $checkpoints/bin/$DATA/512-theta0/400_model.pkl

CUDA_VISIBLE_DEVICES=6 python test.py -data $DATA -r $checkpoints/bin/$DATA/512-theta0.01/400_model.pkl
CUDA_VISIBLE_DEVICES=6 python test.py -data $DATA -r $checkpoints/bin/$DATA/512-theta0.01/500_model.pkl
CUDA_VISIBLE_DEVICES=6 python test.py -data $DATA -r $checkpoints/bin/$DATA/512-theta0.01/600_model.pkl
##CUDA_VISIBLE_DEVICES=5  python pool_test.py -data $DATA -r $checkpoints/bin/$DATA/512/300_model.pkl
#CUDA_VISIBLE_DEVICES=5  python pool_test.py -data $DATA -r $checkpoints/bin/$DATA/512/250_model.pkl
#CUDA_VISIBLE_DEVICES=5  python pool_test.py -data $DATA -r $checkpoints/bin/$DATA/512/450_model.pkl
#CUDA_VISIBLE_DEVICES=5  python pool_test.py -data $DATA -r $checkpoints/bin/$DATA/512/550_model.pkl
#CUDA_VISIBLE_DEVICES=7  python pool_test.py -data $DATA -r $checkpoints/bin/$DATA/512/1200_model.pkl

