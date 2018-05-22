# coding=utf-8
from __future__ import absolute_import, print_function
import argparse
import os
import sys
import torch.utils.data
from torch.backends import cudnn
from torch.autograd import Variable
import models
import losses
from utils import RandomIdentitySampler, mkdir_if_missing, logging, display, orth_reg
import DataSet
import numpy as np
cudnn.benchmark = True


def main(args):

    #  训练日志保存
    log_dir = os.path.join(args.checkpoints, args.log_dir)
    mkdir_if_missing(log_dir)

    sys.stdout = logging.Logger(os.path.join(log_dir, 'log.txt'))
    display(args)

    if args.r is None:
        model = models.create(args.net)
        # load part of the model
        model_dict = model.state_dict()
        # print(model_dict)

        model = models.create(args.net, pretrained=True)
        model.features = torch.nn.Sequential(
            model.features,
            torch.nn.MaxPool2d(7),
            # torch.nn.BatchNorm2d(512),
            torch.nn.Dropout(p=0.01)
        )
        model.classifier = torch.nn.Sequential(
            torch.nn.Linear(512, args.dim)
        )

        # # orth init
        if args.init == 'orth':
            w = model_dict['classifier.0.weight']
            model_dict['classifier.0.weight'] = torch.nn.init.orthogonal_(w)
        else:
            print('initialize the FC layer kaiming-ly')
            w = model_dict['classifier.0.weight']
            model_dict['classifier.0.weight'] = torch.nn.init.kaiming_normal_(w)

        # zero bias
        model_dict['classifier.0.bias'] = torch.zeros(args.dim)
    else:
        # resume model
        print('Resume from model at Epoch %d' % args.start)
        model = torch.load(args.r)

    model = model.cuda()

    torch.save(model, os.path.join(log_dir, 'model.pkl'))
    print('initial model is save at %s' % log_dir)

    # fine tune the model: the learning rate for pre-trained parameter is 1/10
    new_param_ids = set(map(id, model.classifier.parameters()))

    new_params = [p for p in model.parameters() if
                  id(p) in new_param_ids]

    base_params = [p for p in model.parameters() if
                   id(p) not in new_param_ids]
    param_groups = [
                {'params': base_params, 'lr_mult': 0.1},
                {'params': new_params, 'lr_mult': 1.0}]

    optimizer = torch.optim.Adam(param_groups, lr=args.lr,
                                 weight_decay=args.weight_decay)

    if args.loss == 'center-nca':
        criterion = losses.create(args.loss, alpha=args.alpha).cuda()
    elif args.loss == 'cluster-nca':
        criterion = losses.create(args.loss, alpha=args.alpha, beta=args.beta).cuda()
    elif args.loss == 'neighbour':
        criterion = losses.create(args.loss, k=args.k, margin=args.margin).cuda()
    elif args.loss == 'nca':
        criterion = losses.create(args.loss, alpha=args.alpha, k=args.k).cuda()
    elif args.loss == 'triplet':
        criterion = losses.create(args.loss, alpha=args.alpha).cuda()
    elif args.loss == 'bin':
        criterion = losses.create(args.loss, margin=args.margin)
    else:
        criterion = losses.create(args.loss).cuda()

    if args.data == 'shop':
        data = DataSet.create(args.data, root=None, gallery=False, query=False)
    else:
        data = DataSet.create(args.data, root=None, test=False)

    train_loader = torch.utils.data.DataLoader(
        data.train, batch_size=args.BatchSize,
        sampler=RandomIdentitySampler(data.train, num_instances=args.num_instances),
        drop_last=True, num_workers=args.nThreads)

    # save the train information
    epoch_list = list()
    loss_list = list()
    pos_list = list()
    neg_list = list()

    for epoch in range(args.start, args.epochs):
        epoch_list.append(epoch)

        running_loss = 0.0
        running_pos = 0.0
        running_neg = 0.0

        if epoch == 1500:
            optimizer = torch.optim.Adam(param_groups, lr=0.1*args.lr,
                                         weight_decay=args.weight_decay)

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            # wrap them in Variable
            inputs = Variable(inputs.cuda())

            # type of labels is Variable cuda.Longtensor
            labels = Variable(labels).cuda()

            optimizer.zero_grad()

            embed_feat = model(inputs)

            loss, inter_, dist_ap, dist_an = criterion(embed_feat, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_neg += dist_an
            running_pos += dist_ap

            if epoch == 0 and i == 0:
                print(50 * '#')
                print('Train Begin -- HA-HA-HA-HA-AH-AH-AH-AH --')

        loss_list.append(running_loss)
        pos_list.append(running_pos / i)
        neg_list.append(running_neg / i)

        print('[Epoch %05d]\t Loss: %.3f \t Accuracy: %.3f \t Pos-Dist: %.3f \t Neg-Dist: %.3f'
              % (epoch + 1, running_loss, inter_, dist_ap, dist_an))

        if epoch % args.save_step == 0:
            torch.save(model, os.path.join(log_dir, '%d_model.pkl' % epoch))
    np.savez(os.path.join(log_dir, "result.npz"), epoch=epoch_list, loss=loss_list, pos=pos_list, neg=neg_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Metric Learning')

    # hype-parameters
    parser.add_argument('-lr', type=float, default=1e-5, help="learning rate of new parameters")
    parser.add_argument('-BatchSize', '-b', default=128, type=int, metavar='N',
                        help='mini-batch size (1 = pure stochastic) Default: 256')
    parser.add_argument('-num_instances', default=8, type=int, metavar='n',
                        help=' number of samples from one class in mini-batch')
    parser.add_argument('-dim', default=512, type=int, metavar='n',
                        help='dimension of embedding space')
    parser.add_argument('-alpha', default=30, type=int, metavar='n',
                        help='hyper parameter in NCA and its variants')
    parser.add_argument('-beta', default=0.1, type=float, metavar='n',
                        help='hyper parameter in some deep metric loss functions')
    parser.add_argument('-k', default=16, type=int, metavar='n',
                        help='number of neighbour points in KNN')
    parser.add_argument('-margin', default=0.5, type=float,
                        help='margin in loss function')
    parser.add_argument('-init', default='random',
                        help='the initialization way of FC layer')
    parser.add_argument('-orth', default=0, type=float,
                        help='the coefficient orthogonal regularized term')

    # network
    parser.add_argument('-data', default='cub', required=True,
                        help='path to Data Set')
    parser.add_argument('-net', default='bn')
    parser.add_argument('-loss', default='branch', required=True,
                        help='loss for training network')
    parser.add_argument('-epochs', default=600, type=int, metavar='N',
                        help='epochs for training process')
    parser.add_argument('-save_step', default=50, type=int, metavar='N',
                        help='number of epochs to save model')

    # Resume from checkpoint
    parser.add_argument('-r', default=None,
                        help='the path of the pre-trained model')
    parser.add_argument('-start', default=0, type=int,
                        help='resume epoch')

    # basic parameter
    parser.add_argument('-checkpoints', default='/opt/intern/users/xunwang',
                        help='where the trained models save')
    parser.add_argument('-log_dir', default=None,
                        help='where the trained models save')
    parser.add_argument('--nThreads', '-j', default=4, type=int, metavar='N',
                        help='number of data loading threads (default: 2)')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=2e-4)

    main(parser.parse_args())




