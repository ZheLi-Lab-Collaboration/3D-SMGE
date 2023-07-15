# -*- coding: utf-8 -*-
# @Time    : 2022/1/19 22:45
# @Author  : TkiChus
# @Email   : XU.Chao.TkiChus@gmail.com

""" ADMET property prediction in ML """

from rdkit.Chem import AllChem
from tdc.benchmark_group import admet_group
from rdkit import Chem, DataStructs
import numpy as np
from deepforest import CascadeForestRegressor, CascadeForestClassifier
from sklearn.metrics import r2_score, mean_squared_error
from DeepPurpose import utils
from sklearn.svm import SVR, SVC
import pandas as pd
from DeepPurpose.utils import data_process_loader_Property_Prediction

from torch.utils import data
from DeepPurpose.utils import mpnn_collate_func
from torch.utils.data import SequentialSampler
from DeepPurpose.CompoundPred import dgl_collate_func
from skopt import BayesSearchCV

# Bayes
from sklearn.linear_model import BayesianRidge
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import BayesianRidge

import pickle
import os
from sklearn.model_selection import RepeatedKFold, KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score

# the custom function of evaluation score
from Score import ADMET_score



# the folder of save model
save_file_dir = ""
group = admet_group(path="./data")

benchmark = group.get('caco2_wang')
name = benchmark['name']
train_val, test = benchmark['train_val'], benchmark['test']
train, valid = group.get_train_valid_split(benchmark=name, split_type='scaffold', seed=123)


### you can choose the FPS encode or TDC encode
"""---------------------------------FPS encode-----------------------------------------------"""
fps_train_total = []
fps_test_total = []

for i in range(len(train_val)):
    smi_train = (train_val.Drug)[i]
    mol_train = Chem.MolFromSmiles(smi_train)
    fp_train = AllChem.GetMorganFingerprintAsBitVect(mol_train, 2)
    arr_train = np.zeros((1, ))
    DataStructs.ConvertToNumpyArray(fp_train, arr_train)
    print(arr_train)
    fps_train_total.append(arr_train)
fps_train_total = np.asarray(fps_train_total)

for i in range(len(test)):
    smi_test = (test.Drug)[i]
    mol_test = Chem.MolFromSmiles(smi_test)
    fp_test = AllChem.GetMorganFingerprintAsBitVect(mol_test, 2)
    arr_test = np.zeros((1, ))
    DataStructs.ConvertToNumpyArray(fp_test, arr_test)
    print(arr_test)
    fps_test_total.append(arr_test)


fps_test_total = np.asarray(fps_test_total)
print(type(fps_test_total))
print(fps_test_total)
print(fps_test_total.shape)  #(245, 2048)

"-------------------------------TDC encode------------------------------"
# you can choose the encode in "Morgan" or "rdkit_2d_normalized"
drug_encoding = "rdkit_2d_normalized"
train_val_encode_pre = pd.DataFrame(train_val.Drug)
test_encode_pre = pd.DataFrame(test.Drug)
# print(train_val_encode_pre)

train_val_encode = utils.encode_drug(train_val_encode_pre, drug_encoding=drug_encoding, column_name="Drug")
test_encode = utils.encode_drug(test_encode_pre, drug_encoding=drug_encoding, column_name="Drug")


config = utils.generate_config(drug_encoding=drug_encoding,
                               train_epoch=100,
                               LR=0.001,
                               batch_size=1,
                               mpnn_hidden_size=32,
                               mpnn_depth=2,
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

print(train_val_encode["drug_encoding"])
"-----------------------------------------------------------------------------------------------------------"

""" example of find the optimal params"""
max_depth = list(range(20, 201, 10))
n_trees = list(range(100, 801, 10))
n_estimators = list(range(2, 11))

# example of grid parameters, you change it according to
grid_param = {'n_trees': n_trees,
              'max_depth': max_depth,
              'n_estimators': n_estimators,
              'predictor': ['forest', 'xgboost', 'lightgbm']}

k_fold = KFold(n_splits=5)

"""you can choose the optimal model whether classification or regression to train and get the optimal result """
# model = CascadeForestRegressor(random_state=1, max_depth=200, n_trees=50, predictor="xgboost")
# model = GaussianNB()
# model = SVC()
# model = SVR(kernel="poly")
model = BayesianRidge()

### you can get the optimal parameters
# Bayes = BayesSearchCV(model, grid_param, n_iter=10, random_state=14)

### fps_encode train
# model.fit(fps_train_total, train_val.Y)
# pred = model.predict(fps_test_total)


### TDC encode train
model.fit(data_train, train_val.Y)
pred = model.predict(data_test)

"""pickle the model and save model"""
with open(save_file_dir, "wb") as file:
    # if not os.path.exists(save_file_dir):
    #     os.makedirs(save_file_dir)
    pickle.dump(model, file)


""" eval the model"""
r2 = r2_score(test.Y, pred)
error = mean_squared_error(test.Y, pred)
from tdc import Evaluator
evaluator = Evaluator(name='MAE') # Spearman, PR-AUC, ROC-AUC
# y_true: [0.8, 0.7, ...]; y_pred: [0.75, 0.73, ...]
pred = evaluator(test.Y, pred)
print(pred)
print(r2, error)



"""example print the optimal parameters """
# print(Bayes.best_params_)
# #score achieved with best parameter combination
# print(Bayes.best_score_)
#
# #all combinations of hyperparameters
# print(Bayes.cv_results_['params'])
#
# #average scores of cross-validation
# print(Bayes.cv_results_['mean_test_score'])
