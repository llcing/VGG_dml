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
    CUDA_VISIBLE_DEVICES=7   python MCA_train.py -data $DATA  -net vgg  -init orth  -lr 1e-4 -dim $DIM -alpha 16  -k 32   -BatchSize 64 -loss $loss  -epochs 51 -checkpoints $checkpoints -log_dir $loss/$DATA/$DIM  -save_step 100
    Model_LIST="0 100 200 300 400 500"
    for i in $Model_LIST; do
        CUDA_VISIBLE_DEVICES=7  python test.py -data $DATA -r $l/$i$r >>result/$loss/$DATA/$DIM.txt
    done
done
