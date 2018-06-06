#!/usr/bin/env bash
DATA="shop"
loss="bin"
checkpoints="/opt/intern/users/xunwang/checkpoints"
r="_model.pkl"

mkdir $checkpoints
mkdir $checkpoints/$loss/
mkdir $checkpoints/$loss/$DATA/

mkdir result/
mkdir result/$loss/
mkdir result/$loss/$DATA/

DIM="512"

l=$checkpoints/$loss/$DATA/$DIM
mkdir $l
CUDA_VISIBLE_DEVICES=7   python train.py -data $DATA  -BN 0  -init random  -lr 1e-5 -dim $DIM -alpha 40 -num_instances 4 -BatchSize 72 -loss $loss  -epochs 201 -checkpoints $checkpoints  -log_dir $l -save_step 5
Model_LIST="40 60 80 100 120 140 160 180 200"
for i in $Model_LIST; do
    CUDA_VISIBLE_DEVICES=7  python test.py -data $DATA -r $l/$i$r >>result/$loss/$DATA/$DIM.txt
    CUDA_VISIBLE_DEVICES=7  python pool_test.py -data $DATA -r $l/$i$r >>result/$loss/$DATA/$DIM-pool.txt
done


