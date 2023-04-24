# -*- coding: utf-8 -*-
# @Time    : 2022/1/19 22:38
# @Author  : TkiChus
# @Email   : XU.Chao.TkiChus@gmail.com


""" ADMET property prediction in DL"""
""" Classification Task Train """
import pandas as pd
import numpy as np
from DeepPurpose import utils, CompoundPred
from tdc.single_pred import ADME, Tox
from deepforest import CascadeForestClassifier


### origin from of data to get
# data = ADME(name='HIA_Hou').get_data(format='dict')
# X, y = data['Drug'], data['Y']


# DF data show all
pd.set_option('display.max_rows', None)
data = ADME("cyp2d6_veith")
spilt = data.get_split(method='scaffold', frac=[0.7, 0.1, 0.2])
train = spilt.get("train")
val = spilt.get("valid")
test = spilt.get("test")

"""-------------------------------------data encoding-----------------------------------"""
# choose the optimal drug_encoding
drug_encoding = ""
if drug_encoding not in ['RDKit2D', 'Morgan', 'CNN', 'NeuralFP', 'MPNN', 'AttentiveFP', 'GIN_AttrMasking', 'GIN_ContextPred']:
    raise ValueError(
        "You have to specify from 'RDKit2D', 'Morgan', 'CNN', 'NeuralFP', 'MPNN', 'AttentiveFP', 'AttrMasking', 'ContextPred'!")


if drug_encoding == 'RDKit2D':
    drug_encoding = 'rdkit_2d_normalized'

if drug_encoding in ['NeuralFP', 'AttentiveFP', 'GIN_AttrMasking', 'GIN_ContextPred']:
    drug_encoding = 'DGL_' + drug_encoding
#
train_encode = utils.encode_drug(train, drug_encoding=drug_encoding, column_name="Drug")
val_encode = utils.encode_drug(val, drug_encoding=drug_encoding, column_name="Drug")
test_encode = utils.encode_drug(test, drug_encoding=drug_encoding, column_name="Drug")
# train
df_train = pd.DataFrame(train_encode)
df_train = df_train.drop(columns="Drug_ID")
df_train.rename(columns={'Drug': 'SMILES', 'Y': 'Label'}, inplace=True)
# test
df_test = pd.DataFrame(val_encode)
df_test = df_test.drop(columns="Drug_ID")
df_test.rename(columns={'Drug': 'SMILES', 'Y': 'Label'}, inplace=True)
#val
df_val = pd.DataFrame(train_encode)
df_val = df_val.drop(columns="Drug_ID")
df_val.rename(columns={'Drug': 'SMILES', 'Y': 'Label'}, inplace=True)





### model confi, you can change other paramtertes
config = utils.generate_config(drug_encoding=drug_encoding,
                               train_epoch=1500,
                               LR=0.001,
                               batch_size=1024,
                               mpnn_hidden_size=32,
                               mpnn_depth=2,
                               cuda_id='0',
                               result_folder="", # save the loss curve and so on
                               )
### model initialize
model = CompoundPred.model_initialize(**config)
### model train
model.train(df_train, df_val, df_test)
### model save
model.save_model("")

pred = model.predict(df_test)


### eval the model,you can choose the different metrics
from tdc import Evaluator
evaluator = Evaluator(name='PR-AUC') # Spearman, PR-AUC, ROC-AUC
# y_true: [0.8, 0.7, ...]; y_pred: [0.75, 0.73, ...]
metric = evaluator(df_test["Label"], pred)
print("metric", metric)

### convert the result to the 0 or 1
outputs = np.asarray([1 if i else 0 for i in (np.asarray(pred) >= 0.5)])
print(pred)
print(outputs)