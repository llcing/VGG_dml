# coding : utf-8
from __future__ import absolute_import
import heapq
import numpy as np
from utils import to_numpy
import time
import random


def Recall_at_ks(sim_mat, k_s=None, query_ids=None, gallery_ids=None):
    start_time = time.time()
    print(start_time)
    """
    :param sim_mat:
    :param query_ids
    :param gallery_ids

    Compute  [R@1, R@2, R@4, R@8]
    """
    if k_s is None:
        k_s = [1, 2, 4, 8]

    sim_mat = to_numpy(sim_mat)
    m, n = sim_mat.shape
    gallery_ids = np.asarray(gallery_ids)
    if query_ids is None:
        query_ids = gallery_ids
    else:
        query_ids = np.asarray(query_ids)

    num_max = int(1e4)

    if m > num_max:
        samples = list(range(m))
        random.shuffle(samples)
        samples = samples[:num_max]
        sim_mat = sim_mat[samples, :]
        query_ids = [query_ids[k] for k in samples]
        m = num_max

    num_valid = np.zeros(len(k_s))
    for i in range(m):
        if i % 1000 == 0:
            print(i)
        x = sim_mat[i]
        indice = heapq.nlargest(k_s[-1], range(len(x)), x.take)

        if query_ids[i] == gallery_ids[indice[0]]:
            num_valid += 1
            continue

        for k in range(len(k_s) - 1):
            if query_ids[i] in gallery_ids[indice[k_s[k]: k_s[k + 1]]]:
                num_valid[(k + 1):] += 1
                break
    print(time.time())
    t = time.time() - start_time
    print(t)
    return num_valid / float(m)


def Recall_at_ks_products(sim_mat, query_ids=None, gallery_ids=None):
    """
    Compute [R@1, R@10, R@100] for stanford on-line Product
    """
    return Recall_at_ks(sim_mat, query_ids=query_ids, gallery_ids=gallery_ids, k_s=[1, 10, 100])


def Recall_at_ks_shop(sim_mat, query_ids=None, gallery_ids=None):
    """
    Compute [R@1, R@10, R@20, ..., R@50] for In-shop-clothes
    """
    return Recall_at_ks(sim_mat, query_ids=query_ids,
                        gallery_ids=gallery_ids, k_s=[1, 10, 20, 30, 40, 50])


def test():
    import torch
    sim_mat = torch.rand(int(7e4), int(7*400))
    sim_mat = to_numpy(sim_mat)
    query_ids = int(1e4)*list(range(7))
    gallery_ids = int(1e3)*list(range(7))
    print(Recall_at_ks_shop(sim_mat, query_ids, gallery_ids))

if __name__ == '__main__':
    test()
