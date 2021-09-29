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
