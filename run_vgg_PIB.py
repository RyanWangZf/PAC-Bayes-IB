'''train VGG model with PAC-Bayes information bottleneck.
'''
import numpy as np
import torch
import os

from src.dataset import load_data
from src.utils import img_preprocess, setup_seed, predict, eval_metric, feature_map_size
from src.utils import train
from src.models import VGG
from src.pib_utils import train_pib

# __data_set__ = 'cifar10'
# __data_set__ = 'cifar100'
__data_set__ = 'stl10'
# __data_set__ = 'svhn'

__prior_ckpt__ = './checkpoints/{}/vgg_prior.pt'.format(__data_set__)
__save_ckpt__ = './checkpoints/{}/vgg_pib.pt'.format(__data_set__)

opt = {
    'num_epoch':100,
    'batch_size':4, # 32
    'lr':1e-4, 
    'weight_decay':0,
    'beta':1e-1,
    'noise_scale':1e-10,
    'schedule': [50, 80],
    'early_stop': 10,
}

if not os.path.exists('./checkpoints/{}'.format(__data_set__)):
    os.makedirs('./checkpoints/{}'.format(__data_set__))

# set random seed
setup_seed(2020)

# load data & preprocess
x_tr, y_tr, x_va, y_va, x_te, y_te = load_data(__data_set__)

all_tr_idx = np.arange(len(x_tr))
num_class = np.unique(y_va).shape[0]

x_tr, y_tr = img_preprocess(x_tr, y_tr,)
x_va, y_va = img_preprocess(x_va, y_va,)
x_te, y_te = img_preprocess(x_te, y_te,)

# load model
model = VGG(num_classes=num_class, dropout_rate=0.0, last_feature_map_size=feature_map_size(__data_set__))
model.cuda()

# get prior on the validation set
if os.path.exists(__prior_ckpt__):
    print("load prior.")
    model.load_state_dict(torch.load(__prior_ckpt__))
else:
    train(model, np.arange(len(y_va)), x_va, y_va, x_va, y_va, 10, 32, 5e-5, 0, __prior_ckpt__, 5)
w0_dict = dict()
for param in model.named_parameters():
    w0_dict[param[0]] = param[1].clone().detach() # detach but still on gpu
model.w0_dict = w0_dict
model._initialize_weights()
print("done get prior weights")

# start training model
info_dict, loss_acc_dict = train_pib(model, all_tr_idx,
    x_tr, y_tr, x_va, y_va, 
    num_epoch=opt['num_epoch'],
    batch_size=opt['batch_size'],
    lr=opt['lr'],
    weight_decay=opt['weight_decay'],
    beta=opt['beta'],
    early_stop_ckpt_path=__save_ckpt__,
    early_stop_tolerance=opt['early_stop'],
    noise_scale=opt['noise_scale'],
    schedule=opt['schedule'],
    )

# evaluate test acc
pred_te = predict(model, x_te)
acc_te = eval_metric(pred_te, y_te, num_class)
print("test acc:", acc_te)
