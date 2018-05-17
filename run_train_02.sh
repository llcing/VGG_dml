#!/usr/bin/env bash
DATA="cub"
loss="triplet"
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
    l=$checkpoints/$loss/$DATA/$DIM-sgd
    mkdir $checkpoints/$loss/$DATA/$DIM-sgd
    CUDA_VISIBLE_DEVICES=5   python train_sgd.py -data $DATA  -net vgg  -init orth  -lr 1e-2 -dim $DIM -alpha 4  -k 32  -num_instances 2   -BatchSize 32 -loss $loss  -epochs 2001 -checkpoints $checkpoints -log_dir $loss/$DATA/$DIM-sgd  -save_step 200
    Model_LIST="400 800 1200 1600 2000"
    for i in $Model_LIST; do
        CUDA_VISIBLE_DEVICES=5  python test.py -data $DATA -r $l/$i$r >>result/$loss/$DATA/$DIM-sgd.txt
        CUDA_VISIBLE_DEVICES=5  python pool_test.py -data $DATA -r $l/$i$r >>result/$loss/$DATA/$DIM-sgd-pool.txt
    done
done
