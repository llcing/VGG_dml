#!/usr/bin/env bash
DATA="cub"
loss="bin"
checkpoints="/opt/intern/users/xunwang/checkpoints"
r=".pth.tar"
#echo 'Batch Norm before FC test'
mkdir $checkpoints
mkdir $checkpoints/$loss/
mkdir $checkpoints/$loss/$DATA/

mkdir result/
mkdir result/$loss/
mkdir result/$loss/$DATA/


DIM="512"
alpha_list="40"
for alpha in $alpha_list;do
    l=$checkpoints/$loss/$DATA/$DIM-BN-alpha$alpha
    mkdir $l
#    CUDA_VISIBLE_DEVICES=7  python train.py -data $DATA -BN  1  -init random -lr 1e-5 -dim $DIM -alpha $alpha -num_instances 5 -BatchSize 70 -loss $loss  -epochs 401 -checkpoints $checkpoints -save_dir $l  -save_step 50
    Model_LIST="101 201 301 401"
    for i in $Model_LIST; do
        CUDA_VISIBLE_DEVICES=7  python test.py -data $DATA -dim $DIM  -r $l/ckp_ep$i$r >>result/$loss/$DATA/$DIM-BN-alpha$alpha.txt
        CUDA_VISIBLE_DEVICES=7  python pool_test.py -data $DATA -dim $DIM -r $l/ckp_ep$i$r >>result/$loss/$DATA/$DIM-BN-alpha$alpha-pool.txt
    done
done
