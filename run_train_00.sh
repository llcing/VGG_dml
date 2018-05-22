#!/usr/bin/env bash
DATA="shop"
loss="nca"
checkpoints="/opt/intern/users/xunwang/checkpoints"
r="_model.pkl"

mkdir $checkpoints
mkdir $checkpoints/$loss/
mkdir $checkpoints/$loss/$DATA/

mkdir result/
mkdir result/$loss/
mkdir result/$loss/$DATA/

DIM_list="512"
for DIM in $DIM_list;do
    l=$checkpoints/$loss/$DATA/$DIM
    mkdir $checkpoints/$loss/$DATA/$DIM
    CUDA_VISIBLE_DEVICES=7   python train.py -data $DATA  -net vgg  -init random  -lr 1e-5 -dim $DIM -alpha 16 -num_instances 4 -BatchSize 72 -loss $loss  -epochs 141 -checkpoints $checkpoints  -log_dir $loss/$DATA/$DIM -save_step 5
    Model_LIST="20 60 80 100 120 140"
    for i in $Model_LIST; do
        CUDA_VISIBLE_DEVICES=7  python test.py -data $DATA -r $l/$i$r >>result/$loss/$DATA/$DIM.txt
        CUDA_VISIBLE_DEVICES=7  python pool_test.py -data $DATA -r $l/$i$r >>result/$loss/$DATA/$DIM-pool.txt
    done
done
