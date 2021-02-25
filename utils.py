# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd
import pdb
import torchvision.transforms
import torch.optim as optim
import torch.nn.functional as F
from torch import nn

from collections import defaultdict

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def img_preprocess(x, y=None, use_gpu=False):
    x = torch.Tensor(x) / 255.0
    # x = x - 0.5 # [-0.5, 0.5]
    if use_gpu:
        x = x.cuda()

    if y is not None:
        y = torch.LongTensor(y)
        if use_gpu:
            y = y.cuda()

        return x, y

    else:
        return x

def impose_label_noise(y, noise_ratio, return_noise_index=False):
    max_y, min_y = y.max(), y.min()
    all_idx = np.arange(len(y))
    np.random.shuffle(all_idx)
    
    num_noise_sample = int(noise_ratio * len(y))
    noise_idx = all_idx[:num_noise_sample]    
    noise_y = np.random.randint(int(min_y), int(max_y) + 1, [num_noise_sample])
    y[noise_idx] = noise_y
    
    if return_noise_index:
        return y, noise_idx
    else:
        return y

def save_model(ckpt_path, model):
    torch.save(model.state_dict(), ckpt_path)
    return

def load_model(ckpt_path, model):
    try:
        model.load_state_dict(torch.load(ckpt_path))
    except:
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

    return

def predict(model, x):
    model.eval()
    batch_size = 1000
    num_all_batch = np.ceil(len(x)/batch_size).astype(int)
    pred = []
    for i in range(num_all_batch):
        with torch.no_grad():
            pred_ = model(x[i*batch_size:(i+1)*batch_size])
            pred.append(pred_)

    pred_all = torch.cat(pred) # ?, num_class
    return pred_all

def eval_metric(pred, y):
    pred_argmax = torch.max(pred, 1)[1]
    acc = torch.sum((pred_argmax == y).float()) / len(y)
    return acc

def eval_metric_binary(pred, y):
    pred_label = np.ones(len(pred))
    y_label = y.detach().cpu().numpy()
    pred_prob = pred.flatten().cpu().detach().numpy()
    pred_label[pred_prob < 0.5] = 0.0
    acc = torch.Tensor(y_label == pred_label).float().mean()
    return acc

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
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

class ExponentialScheduler(object):
    def __init__(self, init_t, max_t):
        """Args:
            init_t: initial value
            max_t: max value at last
        """
        self.max_t = max_t
        self.init_t = init_t

    def step(self, t):
        return np.minimum(self.max_t, (1.05) ** (t) * self.init_t)

class ExponentialDecayScheduler(object):
    def __init__(self, init_t, min_t):
        """Args:
            init_t: initial value
            max_t: max value at last
        """
        self.min_t = min_t
        self.init_t = init_t

    def step(self, t):
        return np.maximum(self.min_t, (0.95) ** (t) * self.init_t)



class LinearScheduler(object):
    def __init__(self, init_t, max_t, num_epoch):
        self.init_t = init_t
        self.max_t = max_t
        self.num_epoch = num_epoch

    def step(self, t):
        return self.init_t + (self.max_t - self.init_t) / self.num_epoch * t