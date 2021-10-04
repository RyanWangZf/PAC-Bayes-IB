# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from collections import defaultdict
import torch
import numpy as np
from torch import optim
from torch.optim import Optimizer
from torch.autograd import grad

from tqdm import tqdm
import os
import pdb

from .utils import load_model, save_model, eval_metric, predict
from .utils import train

""" Custom optimizer implementations to track various runtime statistics 
refer to https://github.com/noahgolmant/SGLD/blob/eab60b67ff57b182452bc47dd65d2f58b10aabad/sgld/optimizers.py#L7
"""
class SGLD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        noise_scale (float, optional): variance of isotropic noise for langevin
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v
        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
             v = \rho * v + lr * g \\
             p = p - v
        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False,
                 noise_scale=0.1):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGLD, self).__init__(params, defaults)
        self.noise_scale = noise_scale

    def __setstate__(self, state):
        super(SGLD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        returns norm of the step we took for variance analysis later
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)
                p.data.add_(np.sqrt(self.noise_scale), torch.randn_like(p.data))
        return loss

def train_SGLD(model,
    sub_idx,
    x_tr, y_tr, 
    x_va, y_va, 
    num_epoch,
    batch_size,
    lr,
    weight_decay,
    beta,
    early_stop_ckpt_path,
    early_stop_tolerance=3,
    schedule = [50, 80, 100],
    gamma = 0.1,
    noise_scale=1e-4,
    ):
    """Given selected subset, train the model until converge.
    Args:
        model: the trained model class
        sub_idx: picked sample indices in training data
        x_tr, y_tr, x_va, y_va: tr/va data set and labels
        beta: regularization term imposed on I(T;Y) - beta * I(W;S)
        schedule, gamma: on which epoch the learning rate would be shrinked by gamma, e.g., lr = lr * gamma
        noise_scale: the degree of noise in SGLD: w = w  - lr * delta w + noise_scale * N(0,I)
    """
    # early stop
    best_va_acc = 0
    num_all_train = 0
    early_stop_counter = 0
    if early_stop_tolerance < 0:
        early_stop_tolerance = num_epoch

    info_dict = defaultdict(list)
    loss_acc_dict = defaultdict(list)

    # init training with the SGLD optimizer
    optimizer = SGLD(params=filter(lambda p: p.requires_grad, model.parameters()), 
                    lr=lr,
                    momentum=0.9,
                    weight_decay=weight_decay,
                    noise_scale=noise_scale)

    num_all_tr_batch = int(np.ceil(len(sub_idx) / batch_size))

    # num class
    num_class = torch.unique(y_va).shape[0]

    # initialize log p(w) at the first epoch
    energy_decay = 0

    for epoch in tqdm(range(num_epoch)):
        total_loss = 0
        model.train()
        np.random.shuffle(sub_idx)

        # adjust learning rate
        lr = adjust_learning_rate(epoch, optimizer, lr, schedule, gamma)

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

            optimizer.zero_grad()

            if epoch > 0:
                energy_decay.backward(retain_graph=True)
                avg_loss.backward()
            else:
                avg_loss.backward()

            optimizer.step()

            num_all_train += len(x_batch)

            total_loss = total_loss + avg_loss.item()
        
        # compute the information regularization term
        info = model.compute_information_bp_fast(x_tr, y_tr)

        energy_decay = 0
        for k in info.keys():
            # plus decay term for each weight
            energy_decay += info[k]
            info_dict[k].append(info[k].item())
        print("epoch: {}, info: {}".format(epoch, info))
        print("epoch: {}, tr loss: {}, lr: {}, e_decay: {}".format(epoch, total_loss/num_all_tr_batch, lr, energy_decay))
        
        energy_decay = beta * energy_decay

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

            print("epoch: {}, va acc: {}".format(epoch, acc_va.item()))
            loss_acc_dict["va_acc"].append(acc_va.item())

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
        
    return info_dict, loss_acc_dict

def adjust_learning_rate(epoch, optimizer, lr, schedule, decay):
    if epoch in schedule:
        new_lr = lr * decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    else:
        new_lr = lr
    return new_lr

def train_pib(model,
    sub_idx,
    x_tr, y_tr, 
    x_va, y_va, 
    num_epoch,
    batch_size,
    lr,
    weight_decay,
    beta,
    early_stop_ckpt_path,
    early_stop_tolerance=3,
    schedule = [50, 80, 100],
    gamma = 0.1,
    noise_scale=1e-4,
    pretrain_step = 20,
    ):

    if os.path.exists('./checkpoints/vgg_pretrain.pt'):
        model.load_state_dict(torch.load('./checkpoints/vgg_pretrain.pt'))
    else:
        train(model, sub_idx, x_tr, y_tr, x_va, y_va,
            num_epoch=pretrain_step, 
            batch_size=batch_size,
            lr=lr, 
            weight_decay=weight_decay, 
            early_stop_ckpt_path='./checkpoints/vgg_pretrain.pt', 
            early_stop_tolerance=3,
            verbose=False)
    
    info_dict, loss_acc_dict = train_SGLD(model, 
        sub_idx, x_tr, y_tr, x_va, y_va, 
        num_epoch=num_epoch,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        beta=beta,
        early_stop_ckpt_path=early_stop_ckpt_path,
        early_stop_tolerance=early_stop_tolerance,
        noise_scale=noise_scale,
        schedule=schedule,
        gamma=gamma,
        )
    
    # clean the intermediate files
    os.remove('./checkpoints/vgg_pretrain.pt')
    return info_dict, loss_acc_dict

def compute_iiw_bp(model, x_tr, y_tr, param_list, batch_size=1000, no_bp=False):
    '''compute information in weights
    if no_bp set as True, the calculated iiw cannot be used for backward.
    param_list indicates which parameters are used for computing information, e.g.,
    ['extract_feature.0.weight', 'extract_feature.0.bias', 'extract_feature.2.weight', 'extract_feature.2.bias', ...]
    '''
    all_tr_idx = np.arange(len(x_tr))
    np.random.shuffle(all_tr_idx)
    num_all_batch = int(np.ceil(len(x_tr)/batch_size))
    all_model_param_key = [p[0] for p in model.named_parameters()]
    if param_list is None:
        param_list = [p[0] for p in model.named_parameters() if 'weight' in p[0]]
    else:
        # check param list
        for param in param_list:
            if param not in all_model_param_key:
                raise RuntimeError('{} is not found in model parameters!'.format(param))

    delta_w_dict = dict().fromkeys(param_list)
    for pa in model.named_parameters():
        if pa[0] in param_list:
            w0 = model.w0_dict[pa[0]]
            delta_w = pa[1] - w0
            delta_w_dict[pa[0]] = delta_w

    param_ts = [p[1] for p in model.named_parameters() if p[0] in param_list]
    info_dict = dict()
    gw_dict = dict().fromkeys(param_list)

    for _ in range(10):
        sub_idx = np.random.choice(all_tr_idx, batch_size)
        x_batch = x_tr[sub_idx]
        y_batch = y_tr[sub_idx]

        pred = model.forward(x_batch)
        loss = F.cross_entropy(pred, y_batch,
                    reduction="mean")

        gradients = grad(loss, param_ts)        
        for i, gw in enumerate(gradients):
            gw_ = gw.flatten()
            if gw_dict[param_list[i]] is None:
                gw_dict[param_list[i]] = gw_
            else:
                gw_dict[param_list[i]] += gw_
    
    for k in gw_dict.keys():
        gw_dict[k] *= 1/num_all_batch
        delta_w = delta_w_dict[k]
        # delta_w.T gw @ gw.T delta_w = (delta_w.T gw)^2
        info_ = (delta_w.flatten() * gw_dict[k]).sum() ** 2
        if no_bp:
            info_dict[k] = info_.item()
        else:
            info_dict[k] = info_

    return info_dict

def train_iiw(model,
    sub_idx,
    x_tr, y_tr, 
    x_va, y_va, 
    param_list=None,
    num_epoch=100,
    batch_size=32,
    lr=1e-4,
    weight_decay=0,
    beta=1e-1,
    early_stop_ckpt_path='./checkpoints/vgg_pib.pt',
    early_stop_tolerance=10,
    schedule = [50, 80, 100],
    gamma = 0.1,
    noise_scale=1e-10,
    pretrain_step=20,
    verbose=False,
    ):
    '''train model with iiw regularization
    param_list contains the list of parameters which are used to
    compute iiw and regularization. if set None, all parameters
    will be used.
    '''
    # pre-train
    train(model, sub_idx, x_tr, y_tr, x_va, y_va,
        num_epoch=pretrain_step, 
        batch_size=batch_size,
        lr=lr, 
        weight_decay=weight_decay, 
        early_stop_ckpt_path='./checkpoints/iiw_pretrain.pt', 
        early_stop_tolerance=3,
        verbose=False)

    # early stop
    best_va_acc = 0
    num_all_train = 0
    early_stop_counter = 0
    if early_stop_tolerance < 0:
        early_stop_tolerance = num_epoch

    info_dict = defaultdict(list)
    loss_acc_dict = defaultdict(list)

    # init training with the SGLD optimizer
    optimizer = SGLD(params=filter(lambda p: p.requires_grad, model.parameters()), 
                    lr=lr,
                    momentum=0.9,
                    weight_decay=weight_decay,
                    noise_scale=noise_scale)

    num_all_tr_batch = int(np.ceil(len(sub_idx) / batch_size))

    # num class
    num_class = torch.unique(y_va).shape[0]

    # initialize log p(w) at the first epoch
    energy_decay = 0

    for epoch in tqdm(range(num_epoch)):
        total_loss = 0
        model.train()
        np.random.shuffle(sub_idx)

        # adjust learning rate
        lr = adjust_learning_rate(epoch, optimizer, lr, schedule, gamma)

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

            avg_loss = torch.mean(loss)

            optimizer.zero_grad()

            if epoch > 0:
                energy_decay.backward(retain_graph=True)
                avg_loss.backward()
            else:
                avg_loss.backward()

            optimizer.step()
            num_all_train += len(x_batch)
            total_loss = total_loss + avg_loss.item()
        
        # compute the information regularization term
        info = compute_iiw_bp(model, x_tr, y_tr, param_list, no_bp=False)

        energy_decay = 0
        for k in info.keys():
            # plus decay term for each weight
            energy_decay += info[k]
            info_dict[k].append(info[k].item())
        
        if verbose:
            print("epoch: {}, info: {}".format(epoch, info))
            print("epoch: {}, tr loss: {}, lr: {}, e_decay: {}".format(epoch, total_loss/num_all_tr_batch, lr, energy_decay))
        
        energy_decay = beta * energy_decay

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

    os.remove('./checkpoints/iiw_pretrain.pt')
    return info_dict, loss_acc_dict
