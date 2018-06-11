#!/usr/bin/env bash
DATA="cub"
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

lr_list="1e-5"
for lr in $lr_list;do
        l=$checkpoints/$loss/$DATA/Loss$loss-65-lr$lr-net$net
#        resume=$l/350$r
        mkdir $l
#        CUDA_VISIBLE_DEVICES=2   python train.py -data $DATA -net $net  -BN 1  -init random  -lr $lr -num_instances 5 -BatchSize 65 -loss $loss  -epochs 601 -checkpoints $checkpoints  -log_dir $l -save_step 50
        Model_LIST="0 100 200 250 300 350 400"
#        Model_LIST="600 650 700 750 800 900 1000 1200"
        for i in $Model_LIST; do
#            CUDA_VISIBLE_DEVICES=1  python test.py -net $net -data $DATA -batch_size 100 -r $l/$i$r >>result/$loss/$DATA/Loss-$loss-128batchsize-Lr$lr-net$net.txt
            CUDA_VISIBLE_DEVICES=1  python pool_test.py -net $net -data $DATA -batch_size 100 -r $l/$i$r >>result/$loss/$DATA/POOL-Loss-$loss-65batchsize-lr$lr-net$net.txt
        done
done


