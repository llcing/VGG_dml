# Deep Metric Learning

### Learn a deep metric which can be used image retrieval , clustering.
============================

## Pytorch Code for deep metric methods:

- Contrasstive Loss

- Batch-All-Loss and Batch-Hard-Loss

    2 Loss Functions in [In Defense of Triplet Loss in ReID](https://arxiv.org/abs/1703.07737)

- HistogramLoss

    [Learning Deep Embeddings with Histogram Loss](https://arxiv.org/abs/1611.00822)

- BinDevianceLoss

    Self - Modified Version with better performance
    Baseline method in BIER(Deep Metric Learning with BIER: Boosting Independent Embeddings Robustly)

- NCA Loss

   Learning a Nonlinear Embedding by Preserving Class Neighbourhood Structure  -Ruslan Salakhutdinov and Geoffrey Hinton


## Dataset
- [Car-196](http://ai.stanford.edu/~jkrause/cars/car_devkit.tgz)

   first 98 classes as train set and last 98 classes as test set
- [CUB-200-2011](http://www.vision.caltech.edu/visipedia-data/CUB-200/images.tgz)

  first 100 classes as train set and last 100 classes as test set

- [Stanford-Online](ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip)
  
  for the experiments, we split 59,551 images of 11,318 classes for training and 60,502 images of 11,316 classes for testing
  
  - [In-Shop-clothes]

  After downloading all the three data file, you should precess them as above, and put the directionary named DataSet in the project.
  We provide a script to precess CUB( Deep_Metric/DataSet/split_dataset.py ) Car and Stanford online products.

## Pretrained models in Pytorch

Pre-trained VGG-16-BN 

## Prerequisites

- Computer with Linux or OSX
- For training, an NVIDIA GPU is strongly recommended for speed. CPU is supported but training may be slow.
- Pytorch 0.4.0

## Performance of Loss:

To be clear and simple, I only provide Rank@1 on DataSets without test augment. Because, in most case, more higher the Rank@1 is,  more higher the Rank@K.

In_shop_clothes result wil be updated in the near future.

|Loss Function| Rank@1(%)|
|---|---
|BinDeviance Loss|66.5|
|NCA Loss|61.7|

## Reproducing Car-196 (or CUB-200-2011) experiments

**With  BinDeviance Loss  :**

```bash
sh run_train_00.sh
```
