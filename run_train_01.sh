#!/usr/bin/env bash
DATA="cub"
loss="nca"
checkpoints="/opt/intern/users/xunwang/checkpoints"
r="_model.pkl"

mkdir $checkpoints
mkdir $checkpoints/$loss/
mkdir $checkpoints/$loss/$DATA/

mkdir result/
mkdir result/$loss/
mkdir result/$loss/$DATA/

DIM_list="512 64"
for DIM in $DIM_list;do
    l=$checkpoints/$loss/$DATA/$DIM
    mkdir $checkpoints/$loss/$DATA/$DIM
    CUDA_VISIBLE_DEVICES=6   python train.py -data $DATA  -net vgg  -init orth  -lr 1e-5 -dim $DIM -alpha 16  -k 32   -BatchSize 64 -loss $loss  -epochs 301 -checkpoints $checkpoints -log_dir $loss/$DATA/$DIM  -save_step 50
    Model_LIST="0 50 100 150 200 250 300"
    for i in $Model_LIST; do
        CUDA_VISIBLE_DEVICES=6  python test.py -data $DATA -r $l/$i$r >>result/$loss/$DATA/$DIM.txt
    done
done
