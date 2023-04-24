# -*- coding: utf-8 -*-
# @Time    : 2022/1/19 23:07
# @Author  : TkiChus
# @Email   : XU.Chao.TkiChus@gmail.com



""" Prediciton of ADMET in ML"""
import pickle
from tdc.benchmark_group import admet_group
import numpy as np

import pandas as pd
from DeepPurpose.utils import data_process_loader_Property_Prediction

from torch.utils import data
from DeepPurpose.utils import mpnn_collate_func
from torch.utils.data import SequentialSampler
from DeepPurpose.CompoundPred import dgl_collate_func
from sklearn.metrics import r2_score, mean_squared_error
from DeepPurpose import utils


model_file = "best-model/BayesianRidge-caco2_Rmodel/model.pickle"

group = admet_group(path="./data")
predictions_list = []


benchmark = group.get('caco2_wang')

name = benchmark['name']
train_val, test = benchmark['train_val'], benchmark['test']

train, valid = group.get_train_valid_split(benchmark=name, split_type='scaffold', seed=123)
print(train)
print(valid)



### if you choose the FPS encode you should copy the code there
"-------------------------------TDC encode------------------------------"
drug_encoding = "rdkit_2d_normalized"
train_val_encode_pre = pd.DataFrame(train_val.Drug)
test_encode_pre = pd.DataFrame(test.Drug)
# print(train_val_encode_pre)

train_val_encode = utils.encode_drug(train_val_encode_pre, drug_encoding=drug_encoding, column_name="Drug")
test_encode = utils.encode_drug(test_encode_pre, drug_encoding=drug_encoding, column_name="Drug")


config = utils.generate_config(drug_encoding=drug_encoding,
                               cuda_id='0',
                               )

# print(train_val.index.values)

train_val_encode_final = data_process_loader_Property_Prediction(train_val.index.values, train_val.Y.values, train_val_encode, **config)
test_encode_final = data_process_loader_Property_Prediction(test.index.values, test.Y.values, test_encode, **config)



params_train = {'batch_size': config['batch_size'],
					'shuffle': False,
					'num_workers': config['num_workers'],
					'drop_last': False,
					'sampler':SequentialSampler(train_val_encode_final)}

params_test = {'batch_size': config['batch_size'],
					'shuffle': False,
					'num_workers': config['num_workers'],
					'drop_last': False,
					'sampler':SequentialSampler(test_encode_final)}



if (drug_encoding == "MPNN"):
    params_test['collate_fn'] = mpnn_collate_func
elif drug_encoding in ['DGL_GCN', 'DGL_NeuralFP', 'DGL_GIN_AttrMasking', 'DGL_GIN_ContextPred', 'DGL_AttentiveFP']:
    params_test['collate_fn'] = dgl_collate_func
if (drug_encoding == "MPNN"):
    params_train['collate_fn'] = mpnn_collate_func
elif drug_encoding in ['DGL_GCN', 'DGL_NeuralFP', 'DGL_GIN_AttrMasking', 'DGL_GIN_ContextPred', 'DGL_AttentiveFP']:
    params_train['collate_fn'] = dgl_collate_func


train_generator = data.DataLoader(train_val_encode_final, **params_train)
test_generator = data.DataLoader(test_encode_final, **params_test)


data_train = []
data_test = []


for i, (v_d, label) in enumerate(train_generator):
    # print("i", i)
    # print("v_d", v_d)
    v_d = v_d.numpy()
    # print(v_d[0].shape) # [1, 63, 100]
    print(v_d[0].flatten())
    data_train.append(v_d[0].flatten())



data_train = np.asarray(data_train)
print(data_train.shape)

for i, (v_d, label) in enumerate(test_generator):
    # print("i", i)
    # print("v_d", v_d)
    v_d = v_d.numpy()
    # print(v_d[0].shape) # [1, 63, 100]
    # print(v_d[0])
    data_test.append(v_d[0].flatten())
    # print("i"+"v_d", i, v_d)
    # print("label", label)

data_test = np.asarray(data_test)
print(data_test.shape)


print(data_train.shape)

print(type(data_train))
print(train_val_encode["drug_encoding"])
"-----------------------------------------------------------------------------------------------------------"
### load the trained model
with open(model_file, "rb") as file:
    model = pickle.load(file)
    print(model)

### get the prediciton
pred = model.predict(data_test)
print(pred)

### eval
r2 = r2_score(test.Y, pred)
error = mean_squared_error(test.Y, pred)
from tdc import Evaluator
evaluator = Evaluator(name='MAE') # Spearman, PR-AUC, ROC-AUC
# y_true: [0.8, 0.7, ...]; y_pred: [0.75, 0.73, ...]
score = evaluator(test.Y, pred)
print(score)
print(r2, error)

