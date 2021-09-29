import numpy as np
import torch
import os

from src.dataset import load_data
from src.utils import img_preprocess, setup_seed, predict, eval_metric, feature_map_size
from src.utils import train
from src.models import VGG

# __data_set__ = 'cifar10'
# __data_set__ = 'cifar100'
__data_set__ = 'stl10'
# __data_set__ = 'svhn'

__save_ckpt__ = './checkpoints/{}/vgg_vanilla.pt'.format(__data_set__)

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

# start training model
train(model, all_tr_idx, x_tr, y_tr, x_va, y_va,
    num_epoch=20, 
    batch_size=8, # 32
    lr=1e-4, 
    weight_decay=0, 
    early_stop_ckpt_path=__save_ckpt__, 
    early_stop_tolerance=3)

# evaluate test acc
pred_te = predict(model, x_te)
acc_te = eval_metric(pred_te, y_te, num_class)
print("test acc:", acc_te)
