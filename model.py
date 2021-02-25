# -*- coding: utf-8 -*-
import numpy as np
import pdb, os

import torch
import torch.nn.functional as F
from torch import nn
import math
from torch.nn import Parameter

from torch.autograd import grad


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten,self).__init__()

    def forward(self,input):
        return input.view(input.size(0),-1)

class MLP(nn.Module):
    def __init__(self, num_class=10):
        super(MLP, self).__init__()
        self.softmax = nn.functional.softmax

        self.flatten = Flatten()

        self.dense1 = nn.Linear(784,32)
        self.dense2 = nn.Linear(32, num_class)
        self.tanh = nn.Tanh()

        self.early_stopper = None
        self.num_class = num_class

    def compute_information(self, pmat_dict):
        info_dict = dict()
        for pa in self.named_parameters():
            if "weight" in pa[0]:
                "only capture the weight w/o bias info"
                w0 = self.w0_dict[pa[0]]
                delta_w = pa[1].detach().cpu().numpy() - w0
                delta_w = delta_w.reshape(-1,1)
                info_ = delta_w.T @ pmat_dict[pa[0]] @ delta_w
                info_dict[pa[0]] = info_.flatten()
        
        return info_dict

    def compute_emp_fisher_multi(self, x_tr, y_tr, batch_size=1000):
        """Compute empirical fisher information matrix of each parameter in the network.
        """
        def one_hot_transform(y, num_class=10):
            one_hot_y = F.one_hot(y, num_classes=self.num_class)
            return one_hot_y.float()
        self.eval()

        all_tr_idx = np.arange(len(x_tr))
        np.random.shuffle(all_tr_idx)

        num_all_batch = int(np.ceil(len(x_tr)/batch_size))

        pmat = None
        # param_keys = []
        # for pa in self.named_parameters():
        #     if "dense1" not in pa[0]:
        #         param_keys.append(pa[0])

        param_keys = [p[0] for p in self.named_parameters()]
        pmat_dict = dict().fromkeys(param_keys)

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
                gw_ = gw.reshape(-1,1)
                pmat_ = gw_ @ gw_.T
                if pmat_dict[param_keys[i]] is None:
                    pmat_dict[param_keys[i]] = pmat_
                else:
                    pmat_dict[param_keys[i]] += pmat_

        for k in pmat_dict.keys():
            pmat_dict[k] *= 1/num_all_batch
            pmat_dict[k] = pmat_dict[k].cpu().numpy()

        # release GPU Mem occupied by computing emp fisher
        torch.cuda.empty_cache()

        return pmat_dict

    def forward(self, inputs):
        is_training = self.training
        x = self.dense1(inputs)
        x = self.tanh(x)
        x = self.dense2(x)

        out = self.softmax(x, dim=1)

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


class SimpleCNN(nn.Module):
    def __init__(self, num_class=10):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        # self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.maxpool1 = nn.MaxPool2d((2,2))

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d((2,2))
        # self.bn2 = nn.BatchNorm2d(16)

        # name: "dense.weight", "dense.bias"
        self.linear1 = nn.Linear(64 * 8 * 8, 128) # 
        self.linear2 = nn.Linear(128, 128)
        self.dense = nn.Linear(128, num_class)

        self.flatten = Flatten()

        self.softmax = nn.functional.softmax

        self.early_stopper = None

        self._initialize_weights()
        # for name, param in self.named_parameters():
        #     if "weight" in name:
        #         nn.init.normal_(param, mean=0, std=0.01)
        #     if "bias" in name:
        #         nn.init.constant_(param, val=0.01)

    def forward(self, inputs, hidden=False):
        is_training = self.training

        x = self.conv1(inputs) # ?, 64, 32, 32
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x) # ?, 64, 16, 16

        x = self.conv2(x) # ?, 64, 16, 16
        # x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool2(x) # ?, 128, 8, 8
        
        x = self.flatten(x) # ?, 8*8*128

        x = self.linear1(x)
        x = self.relu(x)

        x = self.linear2(x)
        x = self.leaky_relu(x)

        out = self.dense(x) 
        out = self.softmax(out, dim=1)

        if hidden:
            return out, x
        else:
            return out

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

class LeNet(nn.Module):
    def __init__(self, num_class=None, in_size=32, in_channel=3):
        super(LeNet, self).__init__()
        assert num_class != 1 and num_class > 0
        self.conv1 = nn.Conv2d(in_channel, 6, kernel_size=5, stride=1, padding=0)
        self.maxpool1 = nn.MaxPool2d((2,2))
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d((2,2))

        self.flatten = Flatten()

        map_size1 = (in_size - 5) + 1
        map_size2 = ((map_size1 // 2) - 5 + 1) // 2
        
        self.fc1 = nn.Linear(16*map_size2*map_size2, 120)
        self.fc2 = nn.Linear(120, 84)

        # Bayesian linear layer
        if num_class in [None, 2]:
            # binary
            self.dense = BayesLinear(84, 1)

        else:
            # multi-class classification
            self.dense = BayesLinear(84, num_class)

        # custom initialization of weights
        self._initialize_weights()

        self.num_class = num_class

    def encoder(self, inputs):
        x = self.conv1(inputs) # ?, 6, 28, 28
        x = self.relu(x)
        x = self.maxpool1(x) # 10, 6, 14, 14

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2(x) # 10, 16, 5, 5

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x
    
    def forward(self, inputs, hidden=False, sample=False):
        is_training = self.training
        x = self.encoder(inputs)

        # if sample == True, this does Bayesian inference
        out = self.dense(x, sample)

        if self.num_class in [None, 2]:
            # binary classification
            out = torch.sigmoid(out)
        else:
            # multi-class classification
            out = torch.softmax(out, dim=1)

        if hidden:
            return out, x
        else:
            return out
    
    def forward_logits(self, inputs, hidden=False, sample=False):
        """Do forward without softmax/sigmoid output activation.
        """
        is_training = self.training
        x = self.encoder(inputs)

        # if sample == True, this does Bayesian inference
        out = self.dense(x, sample)

        if hidden:
            return out, x
        else:
            return out
    
    def bayes_infer(self, inputs, T=10):
        """Try to do Bayesian inference by Monte Carlo sampling.
        Return:
            pred: T times prediction [T, ?, c]
            epis: epistemic uncertainty of prediction [?, c]
            alea: aleatoric uncertainty of prediction [?, c]
        """
        # the output dense layer must have already been Bayesian
        assert self.dense.eps_distribution is not None
        assert T > 0

        batch_size = inputs.shape[0]

        is_training = self.training

        # deterministic mapping
        x = self.encoder(inputs)

        out = self.dense.bayes_forward(x, T) # T, ?, C

        num_class = out.shape[2]

        # transform to probability distribution
        # &
        # compute uncertainty
        if  num_class == 1:
            # binary classification scenario
            out = torch.sigmoid(out)

            # ---------------------
            # epistemic uncertainty
            # ---------------------
            pbar = out.mean(0)
            temp = out - pbar.unsqueeze(0)
            temp = temp.unsqueeze(3)
            epis = torch.einsum("ijkm,ijnm->ijkn", temp, temp).mean(0) # ?, 1, 1 for binary
            # get diagonal elements of epis
            epis = torch.einsum("ijj->ij", epis) # ?, 1 (?,c for multi-class)

            # ---------------------
            # aleatoric uncertainty
            # ---------------------
            temp = out.unsqueeze(3) # T, ?, c, 1
            phat_op = torch.einsum("ijkm,ijnm->ijkn", temp, temp) # T, ?, c, c
            alea = (out.unsqueeze(2) - phat_op).mean(0).squeeze(2)
            
        else:
            # multi-class
            out = torch.softmax(out, 2) # T, ?, c

            # ---------------------
            # epistemic uncertainty
            # ---------------------
            pbar = out.mean(0) # ?, c
            temp = out - pbar.unsqueeze(0) # T, ?, c
            temp = temp.unsqueeze(3) # T, ?, c, 1
            epis = torch.einsum("ijkm,ijnm->ijkn", temp, temp).mean(0) # ?, c, c

            # get diagonal elements of epis
            epis = torch.einsum("ijj->ij", epis) # ?, c for multi-class

            # ---------------------
            # aleatoric uncertainty
            # ---------------------
            temp = out.unsqueeze(3) # T, ?, c, 1
            phat_op = torch.einsum("ijkm,ijnm->ijkn", temp, temp) # T, ?, c, c
            
            temp = torch.repeat_interleave(temp, num_class , 3) # T, ?, c, c
            eye_mat = torch.eye(num_class).unsqueeze(0).unsqueeze(0).to(self.dense.device) # 1, 1, c, c
            diag_phat = temp * eye_mat
            alea = (diag_phat - phat_op).mean(0) # ?, c, c
            # get diagonal elements of alea
            alea = torch.einsum("ijj->ij", alea) # ?, c for multi-class

        return out, epis, alea

    def set_bayesian_precision(self, precision_matrix):
        self.dense.reset_bayesian_precision_matrix(precision_matrix)
        return

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
            elif isinstance(m, BayesLinear):
                # specifically for the BBBLinear layer
                m.W_mu.data.normal_(0, 0.01)
                m.bias_mu.data.zero_()

class BNN(nn.Module):
    """Deprecated.
    """
    def __init__(self, num_class=None, in_size=32, in_channel=3):
        """If num_class is set "None", it will be a binary classification network.
        """
        super(BNN, self).__init__()
        assert num_class != 1 and num_class > 0

        self.conv1 = nn.Conv2d(in_channel, 32, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.maxpool1 = nn.MaxPool2d((2,2))

        self.flatten = Flatten()

        map_size = in_size // 4
        self.linear1 = nn.Linear(32 * map_size * map_size, 128)

        # Bayesian linear layer
        if num_class in [None, 2]:
            # binary
            self.dense = BayesLinear(128, 1)
            # self.dense = nn.Linear(128, 1)

        else:
            # multi-class classification
            self.dense = BayesLinear(128, num_class)
            # self.dense = nn.Linear(128, num_class)

        self.softmax = nn.functional.softmax

        # custom initialization of weights
        self._initialize_weights()

        self.num_class = num_class

    def encoder(self, inputs):
        x = self.conv1(inputs)
        x = self.relu(x)
        x = self.maxpool1(x) #?, 32, 8, 8
        x = self.flatten(x) # ?, 2048
        x = self.linear1(x) # ?, 128
        x = self.relu(x)
        # x = self.leaky_relu(x)
        return x

    def forward(self, inputs, hidden=False, sample=False):
        is_training = self.training
        x = self.encoder(inputs)

        # if sample == True, this does Bayesian inference
        out = self.dense(x, sample)
        # out = self.dense(x)


        if self.num_class in [None, 2]:
            # binary classification
            out = torch.sigmoid(out)
        else:
            # multi-class classification
            out = self.softmax(out, dim=1)

        if hidden:
            return out, x
        else:
            return out
    
    def forward_logits(self, inputs, hidden=False, sample=False):
        """Do forward without softmax/sigmoid output activation.
        """
        is_training = self.training
        x = self.encoder(inputs)

        # if sample == True, this does Bayesian inference
        out = self.dense(x, sample)
        # out = self.dense(x)


        if hidden:
            return out, x
        else:
            return out
    
    def bayes_infer(self, inputs, T=10):
        """Try to do Bayesian inference by Monte Carlo sampling.
        Return:
            pred: T times prediction [T, ?, c]
            epis: epistemic uncertainty of prediction [?, c]
            alea: aleatoric uncertainty of prediction [?, c]
        """
        # the output dense layer must have already been Bayesian
        assert self.dense.eps_distribution is not None
        assert T > 0

        batch_size = inputs.shape[0]

        is_training = self.training

        # deterministic mapping
        x = self.encoder(inputs)

        out = self.dense.bayes_forward(x, T) # T, ?, C

        num_class = out.shape[2]

        # transform to probability distribution
        # &
        # compute uncertainty
        if  num_class == 1:
            # binary classification scenario
            out = torch.sigmoid(out)

            # ---------------------
            # epistemic uncertainty
            # ---------------------
            pbar = out.mean(0)
            temp = out - pbar.unsqueeze(0)
            temp = temp.unsqueeze(3)
            epis = torch.einsum("ijkm,ijnm->ijkn", temp, temp).mean(0) # ?, 1, 1 for binary
            # get diagonal elements of epis
            epis = torch.einsum("ijj->ij", epis) # ?, 1 (?,c for multi-class)

            # ---------------------
            # aleatoric uncertainty
            # ---------------------
            temp = out.unsqueeze(3) # T, ?, c, 1
            phat_op = torch.einsum("ijkm,ijnm->ijkn", temp, temp) # T, ?, c, c
            alea = (out.unsqueeze(2) - phat_op).mean(0).squeeze(2)
            
        else:
            # multi-class
            out = torch.softmax(out, 2) # T, ?, c

            # ---------------------
            # epistemic uncertainty
            # ---------------------
            pbar = out.mean(0) # ?, c
            temp = out - pbar.unsqueeze(0) # T, ?, c
            temp = temp.unsqueeze(3) # T, ?, c, 1
            epis = torch.einsum("ijkm,ijnm->ijkn", temp, temp).mean(0) # ?, c, c

            # get diagonal elements of epis
            epis = torch.einsum("ijj->ij", epis) # ?, c for multi-class

            # ---------------------
            # aleatoric uncertainty
            # ---------------------
            temp = out.unsqueeze(3) # T, ?, c, 1
            phat_op = torch.einsum("ijkm,ijnm->ijkn", temp, temp) # T, ?, c, c
            
            temp = torch.repeat_interleave(temp, num_class , 3) # T, ?, c, c
            eye_mat = torch.eye(num_class).unsqueeze(0).unsqueeze(0).to(self.dense.device) # 1, 1, c, c
            diag_phat = temp * eye_mat
            alea = (diag_phat - phat_op).mean(0) # ?, c, c
            # get diagonal elements of alea
            alea = torch.einsum("ijj->ij", alea) # ?, c for multi-class

        return out, epis, alea

    def set_bayesian_precision(self, precision_matrix):
        self.dense.reset_bayesian_precision_matrix(precision_matrix)
        return

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
            elif isinstance(m, BayesLinear):
                # specifically for the BBBLinear layer
                m.W_mu.data.normal_(0, 0.01)
                m.bias_mu.data.zero_()


class BayesLinear(nn.Module):
    """Bayesian linear layer in neural networks, here we must predefine the 
    precision matrix for the weight matrix random variable.
    If precision matrix is not given (set as None), 
    this layer will be frequentist (deterministic).
    """
    def __init__(self, in_features, out_features, precision_matrix=None, bias=True):
        super(BayesLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if precision_matrix is not None:
            # sampling for the weight matrix given the precision matrix of Bayesian weight
            num_all_w_param = precision_matrix.shape[0]
            mean_vec = torch.zeros(num_all_w_param).to(self.device)
            self.eps_distribution = torch.distributions.MultivariateNormal(
                loc = mean_vec,
                precision_matrix = precision_matrix)
            print("initialize a Bayesian linear layer!")
        else:
            # now it is a derterministic nn
            self.eps_distribution = None

        # trainable w matrix
        self.W_mu = Parameter(torch.empty((out_features, in_features), device=self.device))

        if self.use_bias:
            self.bias_mu = Parameter(torch.empty((out_features),device=self.device))
        
        else:
            self.register_parameter("bias_mu", None)
        
        self.init_parameters()

    def reset_bayesian_precision_matrix(self, precision_matrix):
        """After frequentist training done, reset the weight to the 
        Bayesian version.
        """
        num_all_w_param = precision_matrix.shape[0]
        mean_vec = torch.zeros(num_all_w_param).to(self.device)
        precision_matrix = precision_matrix.to(self.device)
        self.eps_distribution = torch.distributions.MultivariateNormal(
            loc=mean_vec,
            precision_matrix = precision_matrix)

        print("switch to Bayesian.")

    def init_parameters(self):
        self.W_mu.data.normal_(*(0, 0.1))

        if self.use_bias:
            self.bias_mu.data.normal_(*(0, 0.1))
    
    def forward(self, inputs, sample=False):
        if sample and self.eps_distribution is not None:
            # do inference with uncertainty
            w_eps = self.eps_distribution.sample()
            # random weight
            weight = self.W_mu + w_eps
        
        else:
            # do deterministic mapping with mean weight
            weight = self.W_mu
        
        if self.use_bias:
            bias = self.bias_mu
            out = F.linear(inputs, weight, bias)

        else:
            out = F.linear(inputs, weight, torch.Tensor([0]).to(self.device))
        
        return out
    
    def bayes_forward(self, inputs, T=10):
        """Do parallelled Bayesian inference with multiple weight sampling,
        with efficient einsum realization for tensor operation.
        T, N, D * T, D, C -> T, N, C
        e.g. `res = np.einsum("ijk,ikm->ijm", A, B)`
        """
        w_eps = self.eps_distribution.sample([T]) # (multi) T, c*d; (binary) T, d
        
        # reshape to T, c, d
        w_eps = torch.reshape(w_eps, (T,  self.out_features, w_eps.shape[1] // self.out_features))
        weight = torch.unsqueeze(self.W_mu,0) + w_eps # T, c, d
        weight = torch.einsum("ijk->ikj", weight) # T, d, c

        # copy inputs to T times
        x = torch.unsqueeze(inputs, 0)
        x_T = torch.repeat_interleave(x, T, 0) # T, ?, d

        # do prediction
        pred = torch.einsum("ijk,ikm->ijm",x_T, weight) # T, ?, c

        # plus bias
        if self.use_bias:
            bias = torch.unsqueeze(torch.unsqueeze(self.bias_mu, 0),0) # 1, 1, c
            pred = pred + bias

        return pred

class ResNet(nn.Module):
    """Implementation of a modified ResNet network, on last several layers.
    """
    def __init__(self , model, num_class):
        super(ResNet, self).__init__()
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1]) # drop the lasy layer
        self.trans_layer = nn.Linear(model.fc.in_features, 256) # insert a transition layer
        self.flatten = Flatten()
        self.fc = nn.Linear(256, num_class)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x
    
    def encoder(self, x):
        x = self.resnet_layer(x)
        x = self.flatten(x)
        x = self.trans_layer(x)
        return x        


if __name__ == "__main__":
    lenet = LeNet(10, 64, 3)
    x = torch.randn((10,3,64,64))
    res = lenet(x)
    pdb.set_trace()
    pass
    