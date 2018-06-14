#!/usr/bin/env bash
DATA="shop"
loss="bin"
checkpoints="/opt/intern/users/xunwang/checkpoints"
r="_model.pkl"

mkdir $checkpoints
mkdir $checkpoints/$loss/
mkdir $checkpoints/$loss/$DATA/

mkdir result/
mkdir result/$loss/
mkdir result/$loss/$DATA/


DIM="512"

l=$checkpoints/$loss/$DATA/$DIM-BN
mkdir $l
#CUDA_VISIBLE_DEVICES=4   python train.py -data $DATA  -BN 1  -init random  -lr 1e-5 -dim $DIM -alpha 40 -num_instances 4 -BatchSize 72 -loss $loss  -epochs 161 -checkpoints $checkpoints  -log_dir $l -save_step 5
Model_LIST="140 160"
for i in $Model_LIST; do
    CUDA_VISIBLE_DEVICES=7  python test.py -data $DATA -r $l/$i$r >>result/$loss/$DATA/$DIM-BN.txt
#    CUDA_VISIBLE_DEVICES=4  python pool_test.py -data $DATA -r $l/$i$r >>result/$loss/$DATA/$DIM-BN-pool.txt
done


