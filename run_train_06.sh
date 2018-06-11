#!/usr/bin/env bash
DATA="jd"
loss="dev"
net='vgg'
checkpoints="/opt/intern/users/xunwang/checkpoints"
r="_model.pth"
mkdir $checkpoints
mkdir $checkpoints/$loss/
mkdir $checkpoints/$loss/$DATA/

mkdir result/
mkdir result/$loss/
mkdir result/$loss/$DATA/


lr_list="1e-4"
for lr in $lr_list;do
        l=$checkpoints/$loss/$DATA/Loss$loss-128-lr$lr
        resume=$l/55$r
        mkdir $l
        CUDA_VISIBLE_DEVICES=4,5   python train.py -data $DATA -net $net  -BN 1  -init random  -lr 2e-6 -num_instances 2 -BatchSize 120 -loss $loss -epochs 81 -r $resume -start 55 -checkpoints $checkpoints  -log_dir $l -save_step 5
        Model_LIST="55 60 65 70 75 80"
        for i in $Model_LIST; do
            CUDA_VISIBLE_DEVICES=4,5 python test.py -net $net -data $DATA -batch_size 200 -r $l/$i$r >>result/$loss/$DATA/Loss-$loss-120-batchsize-$lr-lr.txt
        done
done
