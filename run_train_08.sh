#!/usr/bin/env bash
DATA="jd"
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
alpha_list="40"
for alpha in $alpha_list;do
    l=$checkpoints/$loss/$DATA/$DIM-BN-alpha$alpha
#    pre=$l/70_model.pkl
    mkdir $l
    CUDA_VISIBLE_DEVICES=4,5,6   python train.py -data $DATA  -BN 1  -init random  -lr 1e-5 -dim $DIM -alpha $alpha -num_instances 2 -BatchSize 210 -loss $loss  -epochs 161 -checkpoints $checkpoints  -log_dir $l -save_step 5
    Model_LIST="0 5 10 60 80 100 120 140 160"

    for i in $Model_LIST; do
        CUDA_VISIBLE_DEVICES=4  python test.py -data $DATA -r $l/$i$r >>result/$loss/$DATA/$DIM-BN-alpha$alpha.txt
#        CUDA_VISIBLE_DEVICES=1  python pool_test.py -data $DATA -r $l/$i$r >>result/$loss/$DATA/$DIM-BN-alpha$alpha-pool.txt
    done
done


