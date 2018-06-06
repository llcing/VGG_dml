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
alpha_list="40 50"
for alpha in $alpha_list;do
    l=$checkpoints/$loss/$DATA/$DIM-4gpu-alpha$alpha
    mkdir $l
    CUDA_VISIBLE_DEVICES=0,1,6,7  python train.py -data $DATA -BN  1  -init random -lr 3e-5 -dim $DIM -alpha $alpha -num_instances 5 -BatchSize 280 -loss $loss  -epochs 1001 -checkpoints $checkpoints -log_dir $l  -save_step 50
    Model_LIST="400 800 900 1000"
    for i in $Model_LIST; do
#        CUDA_VISIBLE_DEVICES=  python test.py -data $DATA -r $l/$i$r >>result/$loss/$DATA/$DIM-BN-alpha$alpha.txt
        CUDA_VISIBLE_DEVICES=0,1,6,7  python pool_test.py -data $DATA -r $l/$i$r >>result/$loss/$DATA/$DIM-4gpu-alpha$alpha-pool.txt
    done
done

