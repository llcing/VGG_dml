#!/usr/bin/env bash
DATA="car"
loss="bin"
checkpoints="/opt/intern/users/xunwang/checkpoints"
r="_model.pkl"
#echo 'Batch Norm before FC test'
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
    CUDA_VISIBLE_DEVICES=6   python train.py -data $DATA  -net vgg  -init random -lr 1e-5 -dim $DIM  -num_instances 5 -BatchSize 70 -loss $loss  -epochs 1001 -checkpoints $checkpoints -log_dir $l  -save_step 50
    Model_LIST="200 400 600 800 1000"
    for i in $Model_LIST; do
        CUDA_VISIBLE_DEVICES=6  python test.py -data $DATA -r $l/$i$r >>result/$loss/$DATA/$DIM-opt.txt
        CUDA_VISIBLE_DEVICES=6  python pool_test.py -data $DATA -r $l/$i$r >>result/$loss/$DATA/$DIM-opt-pool.txt
    done
done
