# -*- coding: utf-8 -*-
"""Implementation of CL based on transfer learning [1, 2].

[1] Weinshall, D., Cohen, G., & Amir, D. (2018). Curriculum learning by transfer learning: Theory and experiments with deep networks. arXiv preprint arXiv:1802.03796.
[2] Hacohen, G., & Weinshall, D. (2019). On the power of curriculum learning in training deep networks. arXiv preprint arXiv:1904.03626.

"""

import numpy as np
import pdb, os
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from dataset import load_data
from dataset import load_mnist

from utils import setup_seed, img_preprocess, impose_label_noise
from model import MLP
from utils import save_model, load_model
from utils import predict, eval_metric, eval_metric_binary
from config import opt

def train(model,
    sub_idx,
    x_tr, y_tr, 
    x_va, y_va, 
    num_epoch,
    batch_size,
    lr, 
    weight_decay,
    early_stop_ckpt_path,
    early_stop_tolerance=3,
    ):
    """Given selected subset, train the model until converge.
    """
    # early stop
    best_va_acc = 0
    num_all_train = 0
    early_stop_counter = 0

    # init training
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, weight_decay=opt.weight_decay)
    num_all_tr_batch = int(np.ceil(len(sub_idx) / batch_size))

    # num class
    num_class = torch.unique(y_va).shape[0]
    
    for epoch in range(num_epoch):
        total_loss = 0
        model.train()
        np.random.shuffle(sub_idx)

        for idx in range(num_all_tr_batch):
            batch_idx = sub_idx[idx*batch_size:(idx+1)*batch_size]
            x_batch = x_tr[batch_idx]
            y_batch = y_tr[batch_idx]

            pred = model(x_batch)
            if num_class > 2:
                loss = F.cross_entropy(pred, y_batch,
                    reduction="none")
            else:
                loss = F.binary_cross_entropy(pred[:,0], y_batch.float(), 
                    reduction="none")

            sum_loss = torch.sum(loss)
            avg_loss = torch.mean(loss)

            num_all_train += len(x_batch)
            optimizer.zero_grad()
            avg_loss.backward()
            optimizer.step()

            total_loss = total_loss + sum_loss.detach()
        
        # evaluate on va set
        model.eval()
        pred_va = predict(model, x_va)
        if num_class > 2:
            acc_va = eval_metric(pred_va, y_va)
        else:
            acc_va = eval_metric_binary(pred_va, y_va)

        print("epoch: {}, acc: {}".format(epoch, acc_va.item()))
        
        if epoch == 0:
            best_va_acc = acc_va

        if acc_va > best_va_acc:
            best_va_acc = acc_va
            early_stop_counter = 0
            # save model
            save_model(early_stop_ckpt_path, model)

        else:
            early_stop_counter += 1

        if early_stop_counter >= early_stop_tolerance:
            print("early stop on epoch {}, val acc {}".format(epoch, best_va_acc))
            # load model from the best checkpoint
            load_model(early_stop_ckpt_path, model)
            break

    return best_va_acc

def print_cuda_memory():
    t = torch.cuda.get_device_properties(0).total_memory
    c = torch.cuda.memory_cached(0)
    a = torch.cuda.memory_allocated(0)
    f = c-a
    print(f/1024)


def main(**kwargs):
    # pre-setup
    setup_seed(opt.seed)
    opt.parse(kwargs)
    log_dir = os.path.join(opt.result_dir, "vanilla_" + opt.model + "_"  + opt.data_name + "_" + opt.print_opt)
    print("output log dir", log_dir)

    ckpt_dir = os.path.join(os.path.join(log_dir, "ckpt"))
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # intermediate file for early stopping
    early_stop_ckpt_path = os.path.join(ckpt_dir, "best_va.pth")

    output_result_path = os.path.join(log_dir, "vanilla.result")

    # load data & preprocess
    x_tr, y_tr, x_va, y_va, x_te, y_te = load_mnist("./data")
    all_tr_idx = np.arange(len(x_tr))
    num_class = np.unique(y_va).shape[0]

    if opt.noise_ratio > 0:
        print("put noise in label, ratio = ", opt.noise_ratio)
        y_tr = impose_label_noise(y_tr, opt.noise_ratio)
    
    x_tr, y_tr = img_preprocess(x_tr, y_tr, opt.use_gpu)
    x_va, y_va = img_preprocess(x_va, y_va, opt.use_gpu)
    x_te, y_te = img_preprocess(x_te, y_te, opt.use_gpu)
    print("load data done")

    model = MLP()
    if opt.use_gpu:
        model.cuda()
    model.use_gpu = opt.use_gpu

    all_tr_idx = np.arange(len(x_tr))
    
    va_acc = train(model, all_tr_idx, x_tr, y_tr, x_va, y_va, 50, 128, 1e-3, 1e-5, early_stop_ckpt_path, 5)

    # evaluate test acc
    pred_te = predict(model, x_te)
    if num_class > 2:
        acc_te = eval_metric(pred_te, y_te)
    else:
        acc_te = eval_metric_binary(pred_te, y_te)
    print("test acc:", acc_te)

if __name__ == "__main__":
    import fire
    fire.Fire()