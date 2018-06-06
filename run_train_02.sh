#!/usr/bin/env bash
DATA="cub"
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


DIM="512"
alpha_list="20 30 40 50 55 60"
for alpha in $alpha_list;do
    l=$checkpoints/$loss/$DATA/$DIM-BN-alpha$alpha
    mkdir $l
#    CUDA_VISIBLE_DEVICES=6  python train.py -data $DATA -BN  1  -init random -lr 1e-5 -dim $DIM -alpha $alpha -num_instances 5 -BatchSize 70 -loss $loss  -epochs 801 -checkpoints $checkpoints -log_dir $l  -save_step 50
#    Model_LIST="200 300 400 500 600 700 800 900 1000 1200 1400"
    Model_LIST="250 350 450"
    for i in $Model_LIST; do
        CUDA_VISIBLE_DEVICES=6  python test.py -data $DATA -r $l/$i$r >>result/$loss/$DATA/$DIM-BN-alpha$alpha.txt
        CUDA_VISIBLE_DEVICES=6  python pool_test.py -data $DATA -r $l/$i$r >>result/$loss/$DATA/$DIM-BN-alpha$alpha-pool.txt
    done
done
