#!/usr/bin/env bash
DATA="jd"
loss="hdc_bin"
net='hdc'
checkpoints="/opt/intern/users/xunwang/checkpoints"
r="_model.pth"
mkdir $checkpoints
mkdir $checkpoints/$loss/
mkdir $checkpoints/$loss/$DATA/

mkdir result/
mkdir result/$loss/
mkdir result/$loss/$DATA/


alpha_list="40"
for alpha in $alpha_list;do
        l=$checkpoints/$loss/$DATA/1_HDCLoss-128-alpha$alpha
        mkdir $l
#        CUDA_VISIBLE_DEVICES=6,7   python HDC_train.py -data $DATA -net $net  -BN 1  -init random  -lr 1e-5 -alpha $alpha -num_instances 2 -BatchSize 128 -loss $loss -epochs 131 -checkpoints $checkpoints  -log_dir $l -save_step 5
        Model_LIST="5 10 15 20 40 60 80 90 100 110 120 130 140"
        for i in $Model_LIST; do
            CUDA_VISIBLE_DEVICES=4  python test.py -net $net -data $DATA -batch_size 128 -r $l/$i$r >>result/$loss/$DATA/1_HDC_bin-loss-2gpu-128batchsize-alpha$alpha.txt
        done
done



