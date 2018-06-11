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


lr_list="1e-4 1e-5 1e-3"

for lr in $lr_list;do
        l=$checkpoints/$loss/$DATA/HDC-Bin-Loss-128-lr$lr
        mkdir $l
#        CUDA_VISIBLE_DEVICES=4,5   python HDC_train.py -data $DATA -net $net  -BN 1  -init random  -lr $lr -num_instances 2 -BatchSize 128 -loss $loss -epochs 161 -checkpoints $checkpoints  -log_dir $l -save_step 5
        Model_LIST="0 10 20 25 30 40 60 80 90 100 110 120 130 160"
        for i in $Model_LIST; do
            CUDA_VISIBLE_DEVICES=2  python test.py -net $net -data $DATA -batch_size 100 -r $l/$i$r >>result/$loss/$DATA/HDC-Bin-lr$lr-loss-128batchsize.txt
        done
done


