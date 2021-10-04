# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd
import os
import pdb
import torchvision.transforms
import torch.optim as optim
import torch.nn.functional as F
from torch import nn

from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

__LAYER_LIST__ = ['layer 1', 'layer 2', 'layer 3', 'layer 4', 'layer 5']

def img_preprocess(x, y=None, use_gpu=True):
    x = torch.tensor(x) / 255.0
    if use_gpu:
        x = x.cuda()
    if y is not None:
        y = torch.LongTensor(y)
        if use_gpu:
            y = y.cuda()
        return x, y

    else:
        return x

def img_preprocess_cifar(x, y=None, use_gpu=True):
    mean_list = [125.3, 123.0, 113.9]
    std_list = [63.0, 62.1, 66.7]

    new_x_list = []
    for i, m in enumerate(mean_list):
        x_ = (x[:,i] - m) / (std_list[i])
        new_x_list.append(x_)
    
    x = np.array(new_x_list).transpose(1,0,2,3)
    
    # flatten
    x = x.reshape(len(x), 3*32*32)
    x = torch.Tensor(x)

    if use_gpu:
        x = x.cuda()

    if y is not None:
        y = torch.LongTensor(y)
        if use_gpu:
            y = y.cuda()

        return x, y

    else:
        return x

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
    verbose=True,
    ):
    """Given selected subset, train the model until converge.
    """
    # early stop
    best_va_acc = 0
    num_all_train = 0
    early_stop_counter = 0

    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')

    # init training
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    num_all_tr_batch = int(np.ceil(len(sub_idx) / batch_size))

    # num class
    num_class = torch.unique(y_va).shape[0]
    
    for epoch in tqdm(range(num_epoch)):
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

        if x_va is not None:
            # evaluate on va set
            model.eval()
            pred_va = predict(model, x_va)
            acc_va = eval_metric(pred_va, y_va, num_class)
            if verbose:
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
                if verbose:
                    print("early stop on epoch {}, val acc {}".format(epoch, best_va_acc))
                # load model from the best checkpoint
                load_model(early_stop_ckpt_path, model)
                break

    return best_va_acc

def train_prior(model,
    x_tr, y_tr,
    num_epoch=10,
    batch_size=128,
    lr=1e-3,
    weight_decay=1e-5,
    early_stop_ckpt_path="./checkpoints/mlp_prior.pth",
    verbose=False,
    ):
    all_tr_idx = np.arange(len(x_tr))
    train(model, all_tr_idx, x_tr, y_tr, x_tr, y_tr, 
        num_epoch=num_epoch,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        early_stop_ckpt_path=early_stop_ckpt_path,
        verbose=verbose,
        )
    w0_dict = dict()
    for param in model.named_parameters():
        w0_dict[param[0]] = param[1].clone().detach() # detach but still on gpu
    model.w0_dict = w0_dict
    model._initialize_weights()
    print("done get prior weights")

def train_track_info(model,
    sub_idx,
    x_tr, y_tr, 
    x_va, y_va, 
    num_epoch,
    batch_size,
    lr,
    weight_decay,
    track_info_per_iter=-1,
    verbose=True,
    ):
    """Given selected subset, train the model until converge.
    Args:
        model: the trained model class
        sub_idx: picked sample indices in training data
        x_tr, y_tr, x_va, y_va: tr/va data set and labels
        track_info_per_iter: evaluate information per %S iterations (SGD updates),
            if set to -1, track info at the end of every epoch
    """

    info_dict = defaultdict(list)
    loss_acc_dict = defaultdict(list)

    # init training with the SGLD optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

    # num class
    num_class = torch.unique(y_va).shape[0]
    num_all_tr_batch = int(np.ceil(len(sub_idx) / batch_size))
    num_all_train = 0
    iteration = 0
    for epoch in range(num_epoch):
        total_loss = 0
        model.train()
        np.random.shuffle(sub_idx)

        for idx in range(num_all_tr_batch):
            iteration += 1
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

            avg_loss = torch.mean(loss)

            optimizer.zero_grad()

            avg_loss.backward()

            optimizer.step()

            num_all_train += len(x_batch)

            total_loss = total_loss + avg_loss.item()

            if iteration % track_info_per_iter == 0 and track_info_per_iter > 0:
                # estimate information stored in weights
                info = model.compute_information_bp_fast(x_tr, y_tr, no_bp=True)
                for k in info.keys():
                    info_dict[k].append(info[k])
                if verbose:
                    print("iteration/epoch: {}/{}, info: {}".format(iteration, epoch, info))
        if verbose:
            print("epoch: {}, tr loss: {}, lr: {:.6f}".format(epoch, total_loss/num_all_tr_batch, lr))

        # start to evaluate
        if epoch % 1 == 0:
            model.eval()
            pred_tr = predict(model, x_tr)
            acc_tr = eval_metric(pred_tr, y_tr, num_class)

            loss_acc_dict["tr_loss"].append((total_loss/num_all_tr_batch))
            loss_acc_dict["tr_acc"].append(acc_tr.item())

            if x_va is not None:
                # evaluate on va set
                model.eval()
                pred_va = predict(model, x_va)
                acc_va = eval_metric(pred_va, y_va, num_class)
                if verbose:
                    print("epoch: {}, va acc: {}".format(epoch, acc_va.item()))
                loss_acc_dict["va_acc"].append(acc_va.item())
        
        # track info every epoch        
        if track_info_per_iter == -1:
            info = model.compute_information_bp_fast(x_tr, y_tr, no_bp=True)
            for k in info.keys():
                info_dict[k].append(info[k])
            if verbose:
                print("epoch: {}, info: {}".format(epoch, info))
        
            l2_norm = 0
            for pa in model.named_parameters():
                l2_norm += pa[1].data.norm(2)
            loss_acc_dict["l2_norm"].append(l2_norm.cpu().item())



    return info_dict, loss_acc_dict


def save_model(ckpt_path, model):
    torch.save(model.state_dict(), ckpt_path)
    return

def load_model(ckpt_path, model):
    try:
        model.load_state_dict(torch.load(ckpt_path))
    except:
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

    return

def predict(model, x, batch_size=100):
    model.eval()
    num_all_batch = np.ceil(len(x)/batch_size).astype(int)
    pred = []
    for i in range(num_all_batch):
        with torch.no_grad():
            pred_ = model(x[i*batch_size:(i+1)*batch_size])
            pred.append(pred_)

    pred_all = torch.cat(pred) # ?, num_class
    return pred_all

def eval_metric(pred, y, num_class):
    if num_class > 2:
        pred_argmax = torch.max(pred, 1)[1]
        acc = torch.sum((pred_argmax == y).float()) / len(y)
    else:
        acc = eval_metric_binary(pred, y)
    return acc

def eval_metric_binary(pred, y):
    pred_label = np.ones(len(pred))
    y_label = y.detach().cpu().numpy()
    pred_prob = pred.flatten().cpu().detach().numpy()
    pred_label[pred_prob < 0.5] = 0.0
    acc = torch.Tensor(y_label == pred_label).float().mean()
    return acc

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def feature_map_size(dataname):
    ft_map_size = {
        'cifar10':4,
        'cifar100':4,
        'stl10':12,
        'svhn':4,
        }
    return ft_map_size[dataname]


'''specifically used for plot jupyter notebook.
'''
def plot_info_acc(info_dict, loss_acc_list, act, fig_dir='./figure'):
    df_info = pd.DataFrame(info_dict)
    with plt.style.context(['science','nature',]):
        fig, axs = plt.subplots(2, 1, figsize=(6,8))
        for i,col in enumerate(df_info.columns):
            axs[0].plot(df_info[col], label=__LAYER_LIST__[i], lw=2)
        axs[0].set_xlabel('epoch', size=24)
        axs[0].set_ylabel('IIW',size=24)
        axs[0].tick_params(labelsize=20)
        axs[0].set_title('IIW of {} MLP'.format(act), size=20)
        axs[0].legend(fontsize=24)

        # plot loss acc
        ax1 = axs[1]
        ax2 = ax1.twinx()
        lns1 = ax1.plot(loss_acc_list['tr_loss'], label='train loss', color='r', lw=2)
        lns2 = ax2.plot(loss_acc_list['va_acc'], label='test acc', lw=2)
        ax1.set_xlabel('epoch', size=24)
        ax1.set_ylabel('loss', size=24)
        ax2.set_ylabel('acc', size=24)
        ax1.tick_params(labelsize=20)
        ax2.tick_params(labelsize=20)
        ax1.set_ylim(0.3,2.5)
        ax2.set_ylim(0.5,0.8)
        ax1.set_yticks([0.5, 1.0, 1.5, 2.0, 2.5])
        ax2.set_yticks([0.5,0.6,0.7,0.8])
        lns = lns1+lns2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, fontsize=24)
        plt.tight_layout()


    plt.savefig(os.path.join(fig_dir,"{}_acc_loss.png".format(act)),bbox_inches = 'tight')
    plt.show()


def plot_info(info_dict, fig_dir='./figure', use_legend=True):
    '''specifically used for plot jupyter notebook.
    '''
    df_info = pd.DataFrame(info_dict)
    with plt.style.context(['science','nature',]):
        fig, axs = plt.subplots(figsize=(6,4))
        for i,col in enumerate(df_info.columns):
            axs.plot(df_info[col], label=__LAYER_LIST__[i], lw=2)
        axs.set_xlabel('iteration', size=28)
        axs.set_ylabel('IIW',size=28)
        axs.tick_params(labelsize=24)
        axs.yaxis.get_major_formatter().set_powerlimits((0,1))
        axs.set_title('IIW of {}-layer MLP'.format(int(len(df_info.columns))), size=28)
        if use_legend:
            axs.legend(fontsize=26)
        plt.tight_layout()
    plt.savefig(os.path.join(fig_dir,"mlp_{}_info.pdf".format(int(len(df_info.columns)))),bbox_inches = 'tight')
    plt.show()

