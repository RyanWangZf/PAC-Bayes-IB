# -*- coding: utf-8 -*-
import numpy as np
import pdb, os

import torch
import torch.nn.functional as F
from torch import nn
import math
from torch.nn import Parameter

from torch.autograd import grad

class VGG(nn.Module):
    def __init__(self, num_classes, last_feature_map_size=4, dropout_rate=0.0):
        super(VGG, self).__init__()
        self.num_classes = num_classes
        self.num_class = num_classes
        self.last_feature_map_size = last_feature_map_size

        # define an empty for Conv_ReLU_MaxPool
        net = []

        # block 1
        net.append(nn.Conv2d(in_channels=3, out_channels=64, padding=1, kernel_size=3, stride=1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=64, out_channels=64, padding=1, kernel_size=3, stride=1))
        net.append(nn.ReLU())
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # block 2
        net.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
        net.append(nn.ReLU())
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # block 3
        net.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1))
        net.append(nn.ReLU())
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # add net into class property
        self.extract_feature = nn.Sequential(*net)

        # define an empty container for Linear operations
        classifier = []
        classifier.append(nn.Linear(in_features=256*self.last_feature_map_size**2, out_features=1024))
        classifier.append(nn.ReLU())
        classifier.append(nn.Dropout(p=dropout_rate))
        classifier.append(nn.Linear(in_features=1024, out_features=1024))
        classifier.append(nn.ReLU())
        classifier.append(nn.Dropout(p=dropout_rate))
        classifier.append(nn.Linear(in_features=1024, out_features=self.num_classes))

        # add classifier into class property
        self.classifier = nn.Sequential(*classifier)


    def forward(self, x):
        feature = self.extract_feature(x)
        feature = feature.view(x.size(0), -1)

        classify_result = self.classifier(feature)
        return classify_result

    def compute_information_bp_fast(self,  x_tr, y_tr, batch_size=200, no_bp = False):
        """Compute the full information with back propagation support.
        Using delta_w.T gw @ gw.T delta_w = (delta_w.T gw)^2 for efficient computation.
        Args:
            no_bp: detach the information term hence it won't be used for learning.
        """
        def one_hot_transform(y, num_class=100):
            one_hot_y = F.one_hot(y, num_classes=self.num_classes)
            return one_hot_y.float()

        all_tr_idx = np.arange(len(x_tr))
        np.random.shuffle(all_tr_idx)

        num_all_batch = int(np.ceil(len(x_tr)/batch_size))

        param_keys = [p[0] for p in self.named_parameters()]
        delta_w_dict = dict().fromkeys(param_keys)
        for pa in self.named_parameters():
            if "weight" in pa[0]:
                w0 = self.w0_dict[pa[0]]
                delta_w = pa[1] - w0
                delta_w_dict[pa[0]] = delta_w

        info_dict = dict()
        gw_dict = dict().fromkeys(param_keys)

        for idx in range(10):
            # print("compute emp fisher:", idx)
            sub_idx = np.random.choice(all_tr_idx, batch_size)
            x_batch = x_tr[sub_idx]
            y_batch = y_tr[sub_idx]

            y_oh_batch = one_hot_transform(y_batch, self.num_class)
            pred = self.forward(x_batch)
            loss = F.cross_entropy(pred, y_batch,
                        reduction="mean")

            gradients = grad(loss, self.parameters())
            
            for i, gw in enumerate(gradients):
                gw_ = gw.flatten()
                if gw_dict[param_keys[i]] is None:
                    gw_dict[param_keys[i]] = gw_
                else:
                    gw_dict[param_keys[i]] += gw_
        
        for k in gw_dict.keys():
            if "weight" in k:
                gw_dict[k] *= 1/num_all_batch
                delta_w = delta_w_dict[k]
                # delta_w.T gw @ gw.T delta_w = (delta_w.T gw)^2
                info_ = (delta_w.flatten() * gw_dict[k]).sum() ** 2
                if no_bp:
                    info_dict[k] = info_.item()
                else:
                    info_dict[k] = info_

        return info_dict

    def _initialize_weights(self):
        print("initialize model weights.")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class MLP(nn.Module):
    def __init__(self, num_class=10, activation='relu', dropout=0, layers = [512]):
        super(MLP, self).__init__()
        self.softmax = nn.functional.softmax
        self.layers = layers
        self.is_dropout = False
        assert activation in ["linear","tanh","relu","sigmoid"]

        self.flatten = Flatten()
        self.has_hidden = False

        if len(layers) == 0:
            self.mlp_layers = []
            self.dense_final = nn.Linear(28*28, num_class)
        else:
            self.has_hidden = True
            # 728 - 512 - 10
            self.mlp_layers = [nn.Linear(28*28, layers[0])]
            if len(layers) > 1:
                for i, l in enumerate(layers[:-1]):
                    self.mlp_layers.append(nn.Linear(layers[i], layers[i+1]))
                    if activation == "tanh":
                        self.mlp_layers.append(nn.Tanh())
                    elif activation == "relu":
                        self.mlp_layers.append(nn.ReLU())
                    elif activation == "sigmoid":
                        self.mlp_layers.append(nn.Sigmoid())
                    else:
                        self.mlp_layers.append(IdentityMapping())
                    
                    if dropout > 0:
                        self.is_dropout = True
                        self.mlp_layers.append(nn.Dropout(dropout))
            else:
                if activation == "tanh":
                    self.mlp_layers.append(nn.Tanh())
                elif activation == "relu":
                    self.mlp_layers.append(nn.ReLU())
                elif activation == "sigmoid":
                    self.mlp_layers.append(nn.Sigmoid())
                else:
                    self.mlp_layers.append(IdentityMapping())

            self.hidden_net = nn.Sequential(*self.mlp_layers)
            self.dense_final = nn.Linear(layers[-1], num_class)

        self.early_stopper = None
        self.num_class = num_class

    def forward(self, x):
        is_training = self.training
        if self.has_hidden:
            x = self.hidden_net(x)
            x = self.dense_final(x)
        else:
            x = self.dense_final(x)
        return x

    def predict(self, x, batch_size=1000, onehot=False):
        self.eval()
        total_batch = compute_num_batch(x,batch_size)
        pred_result = []

        for idx in range(total_batch):
            batch_x = x[idx*batch_size:(idx+1)*batch_size]
            pred = self.forward(batch_x)
            if onehot:
                pred_result.append(pred)
            else:
                pred_result.append(torch.argmax(pred,1))

        preds = torch.cat(pred_result)
        return preds

    def _initialize_weights(self):
        print("initialize model weights.")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def compute_information_bp_fast(self,  x_tr, y_tr, batch_size=1000, no_bp = False):
        """Compute the full information with back propagation support.
        Using delta_w.T gw @ gw.T delta_w = (delta_w.T gw)^2 for efficient computation.
        Args:
            no_bp: detach the information term hence it won't be used for learning.
        """
        def one_hot_transform(y, num_class=10):
            one_hot_y = F.one_hot(y, num_classes=self.num_class)
            return one_hot_y.float()

        all_tr_idx = np.arange(len(x_tr))
        np.random.shuffle(all_tr_idx)

        num_all_batch = int(np.ceil(len(x_tr)/batch_size))

        param_keys = [p[0] for p in self.named_parameters()]
        delta_w_dict = dict().fromkeys(param_keys)
        for pa in self.named_parameters():
            if "weight" in pa[0]:
                w0 = self.w0_dict[pa[0]]
                delta_w = pa[1] - w0
                delta_w_dict[pa[0]] = delta_w

        info_dict = dict()
        gw_dict = dict().fromkeys(param_keys)

        for idx in range(10):
            # print("compute emp fisher:", idx)
            sub_idx = np.random.choice(all_tr_idx, batch_size)
            x_batch = x_tr[sub_idx]
            y_batch = y_tr[sub_idx]

            y_oh_batch = one_hot_transform(y_batch, self.num_class)
            pred = self.forward(x_batch)
            loss = F.cross_entropy(pred, y_batch,
                        reduction="mean")

            gradients = grad(loss, self.parameters())
            
            for i, gw in enumerate(gradients):
                gw_ = gw.flatten()
                if gw_dict[param_keys[i]] is None:
                    gw_dict[param_keys[i]] = gw_
                else:
                    gw_dict[param_keys[i]] += gw_
        
        for k in gw_dict.keys():
            if "weight" in k:
                gw_dict[k] *= 1/num_all_batch
                delta_w = delta_w_dict[k]
                # delta_w.T gw @ gw.T delta_w = (delta_w.T gw)^2
                info_ = (delta_w.flatten() * gw_dict[k]).sum() ** 2
                if no_bp:
                    info_dict[k] = info_.item()
                else:
                    info_dict[k] = info_

        return info_dict

    def compute_information_bp_fast_all_sample(self,  x_tr, y_tr, batch_size=1000, no_bp = False):
        """Compute the full information with back propagation support.
        Using delta_w.T gw @ gw.T delta_w = (delta_w.T gw)^2 for efficient computation.
        Args:
            no_bp: detach the information term hence it won't be used for learning.
        """
        def one_hot_transform(y, num_class=10):
            one_hot_y = F.one_hot(y, num_classes=self.num_class)
            return one_hot_y.float()

        all_tr_idx = np.arange(len(x_tr))
        np.random.shuffle(all_tr_idx)

        num_all_batch = int(np.ceil(len(x_tr)/batch_size))

        param_keys = [p[0] for p in self.named_parameters()]
        delta_w_dict = dict().fromkeys(param_keys)
        for pa in self.named_parameters():
            if "weight" in pa[0]:
                w0 = self.w0_dict[pa[0]]
                delta_w = pa[1] - w0
                delta_w_dict[pa[0]] = delta_w

        info_dict = dict()
        gw_dict = dict().fromkeys(param_keys)

        for idx in range(num_all_batch):
            # print("compute emp fisher:", idx)
            x_batch = x_tr[batch_size*idx: batch_size*(idx+1)]
            y_batch = y_tr[batch_size*idx: batch_size*(idx+1)]

            y_oh_batch = one_hot_transform(y_batch, self.num_class)
            pred = self.forward(x_batch)
            loss = F.cross_entropy(pred, y_batch,
                        reduction="mean")

            gradients = grad(loss, self.parameters())
            
            for i, gw in enumerate(gradients):
                gw_ = gw.flatten()
                if gw_dict[param_keys[i]] is None:
                    gw_dict[param_keys[i]] = gw_
                else:
                    gw_dict[param_keys[i]] += gw_
        
        for k in gw_dict.keys():
            if "weight" in k:
                gw_dict[k] *= 1/num_all_batch
                delta_w = delta_w_dict[k]
                # delta_w.T gw @ gw.T delta_w = (delta_w.T gw)^2
                info_ = (delta_w.flatten() * gw_dict[k]).sum() ** 2
                if no_bp:
                    info_dict[k] = info_.item()
                else:
                    info_dict[k] = info_

        return info_dict

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten,self).__init__()

    def forward(self,input):
        return input.view(input.size(0),-1)

class IdentityMapping(nn.Module):
    def __init__(self):
        super(IdentityMapping,self).__init__()
    
    def forward(self, input):
        return input

def compute_num_batch(x, batch_size):
    return int(np.ceil(len(x) / batch_size))
