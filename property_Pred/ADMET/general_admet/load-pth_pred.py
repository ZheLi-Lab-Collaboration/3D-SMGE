# -*- coding: utf-8 -*-
# @Time    : 2022/1/19 22:57
# @Author  : TkiChus
# @Email   : XU.Chao.TkiChus@gmail.com

""" Prediciton of ADMET in DL"""
import torch
from DeepPurpose import utils, CompoundPred
from DeepPurpose.CompoundPred import Property_Prediction
import pandas as pd
# import data_merge
import numpy as np

from tdc.single_pred import ADME, Tox

pd.set_option('display.max_rows', None)
### get the datasets
# data = ADME("")
data = Tox("")

spilt = data.get_split(method='scaffold',  seed=1, frac=[0.7, 0.1, 0.2])
train = spilt.get("train")
val = spilt.get("valid")
test = spilt.get("test")

### choose the model drug_encoding
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

print(df_test)


### load your trained model
model = CompoundPred.model_pretrained("best-model/GIN_AttrMasking-ld50_Rmodel")
pred = model.predict(df_test)

### if the task is classification, you should covert the result to 0 or 1
# outputs = np.asarray([1 if i else 0 for i in (np.asarray(pred) >= 0.5)])


### eval the model
from tdc import Evaluator
evaluator = Evaluator(name='MAE')
MAE = evaluator(df_test["Label"], pred)
print("MAE", MAE)
print(df_test)