#!/usr/bin/env bash
DATA="shop"
loss="bin"
net='vgg'
checkpoints="/opt/intern/users/xunwang/checkpoints"
r="_model.pth"
mkdir $checkpoints
mkdir $checkpoints/$loss/
mkdir $checkpoints/$loss/$DATA/

mkdir result/
mkdir result/$loss/
mkdir result/$loss/$DATA/


lr_list="5e-5"
for lr in $lr_list;do
        l=$checkpoints/$loss/$DATA/Loss$loss-120-lr$lr
        resume=$l/35$r
        mkdir $l
        CUDA_VISIBLE_DEVICES=6,7   python train.py -data $DATA -net $net  -BN 1  -init random  -lr 1e-5 -num_instances 5 -BatchSize 120 -loss $loss -epochs 71 -r $resume -start 35 -checkpoints $checkpoints  -log_dir $l -save_step 5
        Model_LIST="35 40 45 50 55 60 65 70"
       for i in $Model_LIST; do
            CUDA_VISIBLE_DEVICES=7  python test.py -net $net -data $DATA -batch_size 100 -r $l/$i$r >>result/$loss/$DATA/2-Loss-$loss-120-batchsize-lr$lr.txt
       done
done

