# Deep Metric Learning

### Learn a deep metric which can be used image retrieval , clustering.
============================

## Pytorch Code for deep metric methods:

- Contrasstive Loss

- Lifted Structure Loss
[](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Song_Deep_Metric_Learning_CVPR_2016_paper.pdf)

    wait to be done in future


- Batch-All-Loss and Batch-Hard-Loss

    2 Loss Functions in [In Defense of Triplet Loss in ReID](https://arxiv.org/abs/1703.07737)


- HistogramLoss

    [Learning Deep Embeddings with Histogram Loss](https://arxiv.org/abs/1611.00822)

- BinDevianceLoss

    Baseline method in BIER(Deep Metric Learning with BIER: Boosting Independent Embeddings Robustly)

- DistWeightDevianceLoss

  My own implement of the sampling way in [sampling matters in deep embedding learning](https://arxiv.org/abs/1706.07567) combined with BinDevianceLoss

  I think my implement is more reasonable and more flexible than the original sampling way in the paper.

- NCA Loss


   Learning a Nonlinear Embedding by Preserving Class Neighbourhood Structure  -Ruslan Salakhutdinov and Geoffrey Hinton


  Though the method was proposed in 2004, It has best performance.


  R@1 is higher 0.61 on  CUB without test augment with Dim 512 finetuned on pretrained inception-v2

- PS: And I have a lot of "wrong" ideas during research the DML problems,
I keep them here without description.
You can see the code by yourself, the code is clear and easy for understanding.
If you have any question about losses that  not been mentioned above,
Feel free to ask me.


## Dataset
- [Car-196](http://ai.stanford.edu/~jkrause/cars/car_devkit.tgz)

   first 98 classes as train set and last 98 classes as test set
- [CUB-200-2011](http://www.vision.caltech.edu/visipedia-data/CUB-200/images.tgz)

  first 100 classes as train set and last 100 classes as test set

- [Stanford-Online](ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip)
  
  for the experiments, we split 59,551 images of 11,318 classes for training and 60,502 images of 11,316 classes for testing

  After downloading all the three data file, you should precess them as above, and put the directionary named DataSet in the project.
  We provide a script to precess CUB( Deep_Metric/DataSet/split_dataset.py ) Car and Stanford online products.

## Pretrained models in Pytorch

Pre-trained Inceptionn-BN(inception-v2) used in most deep metric learning papers

Download site: http://data.lip6.fr/cadene/pretrainedmodels/bn_inception-239d2248.pth

```bash
wget http://data.lip6.fr/cadene/pretrainedmodels/bn_inception-239d2248.pth

mkdir pretrained_models

cp   bn_inception-239d2248.pth    pretrained_models/
```


~~(to save your time, we already download them down and put on my Baidu YunPan.We also put inception v3 in the Baidu YunPan, the performance of inception v-3 is a little worse(about 1.5% on recall@1 ) than inception BN on CUB/Car datasets.)~~
## Prerequisites

- Computer with Linux or OSX
- For training, an NVIDIA GPU is strongly recommended for speed. CPU is supported but training may be slow.

#### Attention!!
The pre-trained model inception-v2 is transferred from Caffe, it can only  work normally on specific version of Pytorch or Python.
I do not figure out why, and do not which version is best(the code can be run without bug, but the performance is bad), but if you want to get similar performance as me
Please create an env as follows:

- Python : 3.5.2 (2.7 may be ok, too)
- [PyTorch](http://pytorch.org)  : (0.2.03)
(I have tried 0.3.0 and 0.1.0,  performance is lower than 0.2.03 by 10% on rank@1)


## Performance of Loss:

To be clear and simple, I only provide Rank@1 on CUB-200 DataSet without test augment. Because, in most case, more higher the Rank@1 is,  more higher the Rank@K.
And better performance on CUB also means better performance on Car-196 , Product-online and other data sets.
If you have finetuned the model to have better performance than below, please tell me, I will update the result here.


|Loss Function| Rank@1(%)|
|---|---
|Pool5-L2|52.4|
|Pool5-512dim L2|49.2|
|Pool5-256dim L2|47.0|
|Pool5-128dim L2|42.0|
|Pool5-64dim L2|32.0|
|Contrastive Loss||
|NeighbourHardLoss||
|NeighbourLoss||
|||
|BinDeviance Loss|51.3|
|HistogramLoss| |
|DistWeightDeviance Loss|51.6|
|SoftmaxNeig Loss|56.3|
|NCA Loss|60.7|

Pool5-512(64, 128, 256)dim L2 means the feature is transformed from Pool5 via a orthogonal transform.

## Reproducing Car-196 (or CUB-200-2011) experiments

**With  NCA Loss  :**

```bash
sh run_train_00.sh
```

To reproduce other experiments, you can edit the run_train.sh file by yourself.

Notice:
the train.py should be modified a little when you used other loss functions.
I will address the problem in these days.


## tSNE visualization on CUB-200
![image](https://github.com/bnulihaixia/Deep_metric/blob/master/Vision/tsne-cub.jpg)
