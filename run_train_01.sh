#!/usr/bin/env bash
DATA="cub"
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

#Model_LIST="20 40 60 80 100"
lr_list="1e-4"
for lr in $lr_list;do
        l=$checkpoints/$loss/$DATA/Loss$loss-128-lr$lr
        mkdir $l
#        CUDA_VISIBLE_DEVICES=1 python train.py -data $DATA -net $net  -BN 1  -init random  -lr $lr -num_instances 5 -BatchSize 65 -loss $loss -epochs 801 -checkpoints $checkpoints  -log_dir $l -save_step 50
#        Model_LIST="50 100 200 300 350 400 450 500 550 600 650 700 800"
        Model_LIST="50 100 150 200 250 300"
        for i in $Model_LIST; do
            CUDA_VISIBLE_DEVICES=0  python test.py -net $net -data $DATA -batch_size 60 -r $l/$i$r >>result/$loss/$DATA/Loss-$loss-65batchsize-lr$lr.txt
            CUDA_VISIBLE_DEVICES=0 python pool_test.py -net $net -data $DATA -batch_size 60 -r $l/$i$r >>result/$loss/$DATA/POOL-Loss-$loss-65batchsize-lr$lr.txt
        done
done
