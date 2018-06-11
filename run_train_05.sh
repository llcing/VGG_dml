#!/usr/bin/env bash
DATA="shop"
loss="bin"
net='vgg16'
checkpoints="/opt/intern/users/xunwang/checkpoints"
r="_model.pth"
mkdir $checkpoints
mkdir $checkpoints/$loss/
mkdir $checkpoints/$loss/$DATA/

mkdir result/
mkdir result/$loss/
mkdir result/$loss/$DATA/


lr_list="2e-5"
for lr in $lr_list;do
        l=$checkpoints/$loss/$DATA/Loss$loss-120-lr$lr-net-$net-dim-384
        mkdir $l
        CUDA_VISIBLE_DEVICES=6,7   python train.py -data $DATA -net $net  -BN 1 -dim 384  -init random  -lr 1e-5 -num_instances 5 -BatchSize 120 -loss $loss -epochs 71 -checkpoints $checkpoints  -log_dir $l -save_step 5
#        Model_LIST="35 40 45 50 55 60 65 70"
       Model_LIST="10 20 30 40 50 60 70"
       for i in $Model_LIST; do
            CUDA_VISIBLE_DEVICES=7  python test.py -net $net -data $DATA -batch_size 100 -r $l/$i$r >>result/$loss/$DATA/2-Loss-$loss-120-batchsize-lr$lr-net-$net-dim-384.txt
       done
done

