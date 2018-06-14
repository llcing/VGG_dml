#!/usr/bin/env bash
DATA="jd"
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
#    CUDA_VISIBLE_DEVICES=4,5,6   python train.py -data $DATA  -BN 1  -init random  -lr 1e-5 -dim $DIM -alpha $alpha -num_instances 2 -BatchSize 210 -loss $loss  -epochs 11  -save_dir $l -save_step 5
    Model_LIST='10'
    for i in $Model_LIST; do
        CUDA_VISIBLE_DEVICES=4  python test.py -data $DATA -r $l/ckp_ep$i$r >>result/$loss/$DATA/$DIM-BN-alpha$alpha.txt
    done
done


