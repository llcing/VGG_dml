#!/usr/bin/env bash
DATA="shop"
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
#    CUDA_VISIBLE_DEVICES=7   python train.py -data $DATA  -net vgg  -init random -lr 1e-5 -dim $DIM  -num_instances 5 -BatchSize 70 -loss $loss  -epochs 2001 -checkpoints $checkpoints -log_dir $loss/$DATA/$DIM  -save_step 50
    Model_LIST="900 950 1000 1050 1100 1200 1400 1600  1800 1900 2000"
    for i in $Model_LIST; do
        CUDA_VISIBLE_DEVICES=7  python test.py -data $DATA -r $l/$i$r >>result/$loss/$DATA/$DIM-.txt
        CUDA_VISIBLE_DEVICES=7  python pool_test.py -data $DATA -r $l/$i$r >>result/$loss/$DATA/$DIM--pool.txt
    done
done
