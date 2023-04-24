# -*- coding: utf-8 -*-
# @Time    : 2022/1/3 11:21
# @Author  : TkiChus
# @Email   : XU.Chao.TkiChus@gmail.com

import csv
import pandas as pd
import numpy as np
from DeepPurpose import utils
from tdc.benchmark_group import admet_group

from DeepPurpose.utils import data_process_loader_Property_Prediction

from torch.utils import data
from DeepPurpose.utils import mpnn_collate_func
from torch.utils.data import SequentialSampler
from DeepPurpose.CompoundPred import dgl_collate_func

def create_csv(path):
    headers = ["SMILES", "Label", "drug_encoding", "Score"]
    with open(path, "w", newline="") as f:
        csv_write = csv.writer(f)
        csv_write.writerow(headers)

def writer(path):
    with open(path, "r") as f:
        csv_reader = csv.reader(f)
        return csv_reader

def clo_add(path, col_name, add_data):
    """
    Args:
        path: your col path
        col_name: specific col
        add_data: add data

    Returns:

    """
    data = pd.read_csv(path, header=True, names=['SMILES', 'Label', "Drug_Encoding"])
    data[col_name] = add_data
    data.to_csv(path, mode='a', header=None, index=0)
    print("----add succeed!!!")


def get_merge_encoding(test_smile_path, drug_encoding):
    data = pd.read_csv(test_smile_path)
    data_smiles = data["SMILES"]
    data_smiles = pd.DataFrame(data_smiles)
    data_encoding = utils.encode_drug(data_smiles, drug_encoding=drug_encoding)
    data_encoding_single = data_encoding["drug_encoding"]
    data["drug_encoding"] = data_encoding_single
    return data


def smiles_encode(data_smiles, drug_encoding):
    if drug_encoding not in ['RDKit2D', 'Morgan', 'CNN', 'NeuralFP', 'MPNN', 'AttentiveFP', 'GIN_AttrMasking',
                             'GIN_ContextPred']:
        raise ValueError(
            "You have to specify from 'RDKit2D', 'Morgan', 'CNN', 'NeuralFP', 'MPNN', 'AttentiveFP', 'AttrMasking', 'ContextPred'!")

    if drug_encoding == 'RDKit2D':
        drug_encoding = 'rdkit_2d_normalized'

    if drug_encoding in ['NeuralFP', 'AttentiveFP', 'GIN_AttrMasking', 'GIN_ContextPred']:
        drug_encoding = 'DGL_' + drug_encoding

    data_smiles = pd.DataFrame(data_smiles)
    data_encoding = utils.encode_drug(data_smiles, drug_encoding=drug_encoding)
    data_encoding_single = data_encoding["drug_encoding"]
    return data_encoding_single

def smiles_ml_encode(test, smiles, drug_encoding):
    # group = admet_group(path="../data")
    #
    # benchmark = group.get(dataset)
    # # print(benchmark)
    # # predicitions = {}
    # name = benchmark['name']
    # train_val, test = benchmark['train_val'], benchmark['test']
    # # print(train_val)
    # train, valid = group.get_train_valid_split(benchmark=name, split_type='scaffold', seed=123)


    "-------------------------------TDC.rdkit_2d encode------------------------------"
    drug_encoding = "rdkit_2d_normalized"
    # train_val_encode_pre = pd.DataFrame(train_val.Drug)
    test_encode_pre = pd.DataFrame(smiles)
    # print(train_val_encode_pre)

    # train_val_encode = utils.encode_drug(train_val_encode_pre, drug_encoding=drug_encoding, column_name="Drug")
    test_encode = utils.encode_drug(test_encode_pre, drug_encoding=drug_encoding, column_name="SMILES")

    config = utils.generate_config(drug_encoding=drug_encoding,
                                   train_epoch=100,
                                   LR=0.001,
                                   batch_size=1,
                                   mpnn_hidden_size=32,
                                   mpnn_depth=2,
                                   cuda_id='0',
                                   )



    # train_val_encode_final = data_process_loader_Property_Prediction(train_val.index.values, train_val.Y.values,
    #                                                                  train_val_encode, **config)
    test_encode_final = data_process_loader_Property_Prediction(test.index.values, test.Label.values, test_encode, **config)

    # params_train = {'batch_size': config['batch_size'],
    #                 'shuffle': False,
    #                 'num_workers': config['num_workers'],
    #                 'drop_last': False,
    #                 'sampler': SequentialSampler(train_val_encode_final)}

    params_test = {'batch_size': config['batch_size'],
                   'shuffle': False,
                   'num_workers': config['num_workers'],
                   'drop_last': False,
                   'sampler': SequentialSampler(test_encode_final)}

    if (drug_encoding == "MPNN"):
        params_test['collate_fn'] = mpnn_collate_func
    elif drug_encoding in ['DGL_GCN', 'DGL_NeuralFP', 'DGL_GIN_AttrMasking', 'DGL_GIN_ContextPred', 'DGL_AttentiveFP']:
        params_test['collate_fn'] = dgl_collate_func
    # if (drug_encoding == "MPNN"):
    #     params_train['collate_fn'] = mpnn_collate_func
    # elif drug_encoding in ['DGL_GCN', 'DGL_NeuralFP', 'DGL_GIN_AttrMasking', 'DGL_GIN_ContextPred', 'DGL_AttentiveFP']:
    #     params_train['collate_fn'] = dgl_collate_func

    # train_generator = data.DataLoader(train_val_encode_final, **params_train)
    test_generator = data.DataLoader(test_encode_final, **params_test)

    # data_train = []
    data_test = []

    # for i, (v_d, label) in enumerate(train_generator):
    #     # print("i", i)
    #     # print("v_d", v_d)
    #     v_d = v_d.numpy()
    #     # print(v_d[0].shape) # [1, 63, 100]
    #     # print(v_d[0].flatten())
    #     data_train.append(v_d[0].flatten())
    #
    # data_train = np.asarray(data_train)
    # print(data_train.shape)

    for i, (v_d, label) in enumerate(test_generator):
        # print("i", i)
        # print("v_d", v_d)
        v_d = v_d.numpy()
        # print(v_d[0].shape) # [1, 63, 100]
        # print(v_d[0])
        # flatten feature
        data_test.append(v_d[0].flatten())
        # print("i"+"v_d", i, v_d)
        # print("label", label)

    data_test = np.asarray(data_test)
    # print(data_test.shape)

    # print(data_train.shape)

    # print(type(data_train))
    # print(train_val_encode["drug_encoding"])
    return data_test

# convert .smi->.csv transit into admet predict
def general_admet_col(uniqueness_smi, csv_path):
    # create_csv(csv_path)
    headers = ["SMILES", "Label", "drug_encoding"]
    with open(uniqueness_smi, "r") as ur:
        smiles_lines = ur.readlines()
        print(smiles_lines)
        with open(csv_path, "a", newline="", encoding="utf8") as f:
            csv_write = csv.writer(f, quoting=csv.QUOTE_NONE, quotechar=None)
            csv_write.writerow(headers)
            for i in range(len(smiles_lines)):
                # smi_label = pd.DataFrame([[smiles_lines[i], i]], columns=["SMILES", "Label"])
                csv_write.writerow([smiles_lines[i].strip("\n"), i])
    ur.close()
    f.close()


if __name__ == '__main__':
    general_admet_col("../data/mode2_200.smi", "../data/mode2_200.csv")













