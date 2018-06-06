#!/usr/bin/env bash
DATA="jd"
loss="bin"
checkpoints="/opt/intern/users/xunwang/checkpoints"
r="_model.pkl"
echo 'Fine-tune on In-shop-clothes'
mkdir $checkpoints
mkdir $checkpoints/$loss/
mkdir $checkpoints/$loss/$DATA/

mkdir result/
mkdir result/$loss/
mkdir result/$loss/$DATA/


DIM="512"
#r='/opt/intern/users/xunwang/checkpoints/bin/shop/512-BN-alpha20/200_model.pkl'
alpha_list="40"

for alpha in $alpha_list;do
    l=$checkpoints/$loss/$DATA/$DIM-fine-alpha$alpha
    mkdir $l
#    CUDA_VISIBLE_DEVICES=5   python train.py -data $DATA  -BN 1  -init random  -lr 1e-5 -dim $DIM -alpha $alpha -num_instances 2 -BatchSize 70 -loss $loss  -epochs 151 -r $r -checkpoints $checkpoints  -log_dir $l -save_step 5
    Model_LIST="5"

    for i in $Model_LIST; do
        CUDA_VISIBLE_DEVICES=5  python test.py -data $DATA -r $l/$i$r >>result/$loss/$DATA/$DIM-fine-alpha$alpha.txt
#        CUDA_VISIBLE_DEVICES=1  python pool_test.py -data $DATA -r $l/$i$r >>result/$loss/$DATA/$DIM-BN-alpha$alpha-pool.txt
    done
done