# -*- coding: utf-8 -*-
# @Time    : 2022/1/19 23:14
# @Author  : TkiChus
# @Email   : XU.Chao.TkiChus@gmail.com

""" all ADMET Prediction"""

import torch
import pandas as pd
from DeepPurpose import utils
import data_merge
import pickle
import os
from DeepPurpose import CompoundPred

import numpy as np

import argparse

parser = argparse.ArgumentParser()


parser.add_argument('--smi_path', type=str, help='Path to directory containing smi_files to convert into csv format.',
                    default='../data/smi_files/')

parser.add_argument('--csv_path', type=str, help='Path to directory containing csv_files to put into ADMET prediction.'
                                                 'The generated molecules of SMILES format are converted into csv format',
                    default='../data/csv_files/')
parser.add_argument('--admet_result_path', type=str, help='Path to directory containing the result of final ADMET prediction.',
                    default='../data/csv_files/')

args = parser.parse_args()



data_merge.general_admet_col(args.smi_path, args.csv_path)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
### data is used to add the drug_encoding data into predict, which is an intermediate variable
data = pd.read_csv(args.csv_path)
### The data_add_score here is used to add all the predictions to the original one, which is actually an intermediate variable
data_add_score = data

data_smiles = data["SMILES"]

data_smiles = pd.DataFrame(data_smiles)


print(data)
print(data_add_score)

"""--------------------------DL9 encode------------------------"""
"""A：Absorption"""
### Caco2 prediction
# get the encode smiles
caco2_drug_encoding = "rdkit_2d_normalized"
caco2_test_data = data_merge.smiles_ml_encode(data, data_smiles, caco2_drug_encoding)
# predict the result and add the pred to df
with open("../best-model/BayesianRidge-caco2_Rmodel/model.pickle", "rb") as file:
    caco2_model = pickle.load(file)
    print(caco2_model)

print(caco2_test_data)
caco2_pred = caco2_model.predict(caco2_test_data)
data_add_score["Caco2_pred"] = caco2_pred
print(data_add_score)


### HIA prediciton
# get the HIA encode
hia_drug_encoding = "GIN_AttrMasking"
hia_data_encoding_single = data_merge.smiles_encode(data_smiles, hia_drug_encoding)
data["drug_encoding"] = hia_data_encoding_single
# predict the result and add the pred to df
hia_model = CompoundPred.model_pretrained("../best-model/GIN_AttrMasking-HIA_Hou_Cmodel")
hia_pred = hia_model.predict(data)
hia_pred_binary = np.asarray([1 if i else 0 for i in (np.asarray(hia_pred) >= 0.5)])
data_add_score["HIA_pred"] = hia_pred_binary
print(data_add_score)


### Pgp prediciton
# get the Pgp encode
pgp_drug_encoding = "GIN_AttrMasking"
pgp_data_encoding_single = data_merge.smiles_encode(data_smiles, pgp_drug_encoding)
data["drug_encoding"] = pgp_data_encoding_single
# predict the result and add the pred to df
pgp_model = CompoundPred.model_pretrained("../best-model/GIN_AttrMasking-Pgp_Cmodel")
pgp_pred = pgp_model.predict(data)
pgp_pred_binary = np.asarray([1 if i else 0 for i in (np.asarray(pgp_pred) >= 0.5)])
data_add_score["Pgp_pred"] = pgp_pred_binary
print(data_add_score)

### Bioav
# get the bioav encode
bioav_drug_encoding = "RDKit2D"
bioav_data_encoding_single = data_merge.smiles_encode(data_smiles, bioav_drug_encoding)
data["drug_encoding"] = bioav_data_encoding_single
# predict the result and add the pred to df
bioav_model = CompoundPred.model_pretrained("../best-model/RDKit2D-Bioav_Cmodel")
bioav_pred = bioav_model.predict(data)
bioav_pred_binary = np.asarray([1 if i else 0 for i in (np.asarray(bioav_pred) >= 0.5)])
data_add_score["Bioav_pred"] = bioav_pred_binary
print(data_add_score)


### Lipo
# get the lipo encode
lipo_drug_encoding = "GIN_ContextPred"
lipo_data_encoding_single = data_merge.smiles_encode(data_smiles, lipo_drug_encoding)
data["drug_encoding"] = lipo_data_encoding_single
# predict the result and add the pred to df
lipo_model = CompoundPred.model_pretrained("../best-model/GIN_ContextPred-Lipo_Rmodel")
lipo_pred = lipo_model.predict(data)
data_add_score["Lipo_pred"] = lipo_pred
print(data_add_score)


### AqSol
# get the aqsol encode
aqsol_drug_encoding = "AttentiveFP"
aqsol_data_encoding_single = data_merge.smiles_encode(data_smiles, aqsol_drug_encoding)
data["drug_encoding"] = aqsol_data_encoding_single
# predict the result and add the pred to df
aqsol_model = CompoundPred.model_pretrained("../best-model/AttentiveFP-AqSol_Rmodel")
aqsol_pred = aqsol_model.predict(data)
data_add_score["AqSol_pred"] = aqsol_pred
print(data_add_score)


"""D: Distribution"""
### BBB
# get the bbb model encode
bbb_drug_encoding = "GIN_ContextPred"
bbb_data_encoding_single = data_merge.smiles_encode(data_smiles, bbb_drug_encoding)
data["drug_encoding"] = bbb_data_encoding_single
# predict the result and add the pred to df
bbb_model = CompoundPred.model_pretrained("../best-model/GIN_ContextPred-BBB_Cmodel")
bbb_pred = bbb_model.predict(data)
bbb_pred_binary = np.asarray([1 if i else 0 for i in (np.asarray(bbb_pred) >= 0.5)])
data_add_score["BBB_pred"] = bbb_pred_binary
print(data_add_score)


### PPBR
# get the ppbr model encode
ppbr_drug_encoding = "rdkit_2d_normalized"
ppbr_test_data = data_merge.smiles_ml_encode(data, data_smiles, ppbr_drug_encoding)
# predict the result and add the pred to df
with open("../best-model/SVR_PPBR_Rmodel/model.pickle", "rb") as file:
    ppbr_model = pickle.load(file)
    print(ppbr_model)
print(ppbr_test_data)
ppbr_pred = ppbr_model.predict(ppbr_test_data)
data_add_score["PPBR_pred"] = ppbr_pred
print(data_add_score)





### VD
# get the encode smiles
vd_drug_encoding = "rdkit_2d_normalized"
vd_test_data = data_merge.smiles_ml_encode(data, data_smiles, vd_drug_encoding)
# predict the result and add the pred to df
with open("../best-model/SVR-VD_Rmodel/model.pickle", "rb") as file:
    vd_model = pickle.load(file)
    print(vd_model)

# print(vd_test_data)
vd_pred = vd_model.predict(vd_test_data)
data_add_score["VD_pred"] = vd_pred
print(data_add_score)


"""M：Metabolism"""
### CYP2D6-I
# get the cyp2d6-i encode
cyp2d6i_drug_encoding = "GIN_ContextPred"
cyp2d6i_data_encoding_single = data_merge.smiles_encode(data_smiles, cyp2d6i_drug_encoding)
data["drug_encoding"] = cyp2d6i_data_encoding_single
# predict the result and add the pred to df
cyp2d6i_model = CompoundPred.model_pretrained("../best-model/GIN_ContextPred-CYP2D6-I_Cmodel")
cyp2d6i_pred = cyp2d6i_model.predict(data)
cyp2d6i_pred_binary = np.asarray([1 if i else 0 for i in (np.asarray(cyp2d6i_pred) >= 0.5)])
data_add_score["CYP2D6-I_pred"] = cyp2d6i_pred_binary
print(data_add_score)


### CYP3A4-I
# get the cyp3a4-i encode
cyp3a4i_drug_encoding = "GIN_ContextPred"
cyp3a4i_data_encoding_single = data_merge.smiles_encode(data_smiles, cyp3a4i_drug_encoding)
data["drug_encoding"] = cyp3a4i_data_encoding_single
# predict the result and add the pred to df
cyp3a4i_model = CompoundPred.model_pretrained("../best-model/GIN_ContextPred-CYP3A4-I_Cmodel")
cyp3a4i_pred = cyp3a4i_model.predict(data)
cyp3a4i_pred_binary = np.asarray([1 if i else 0 for i in (np.asarray(cyp3a4i_pred) >= 0.5)])
data_add_score["CYP3A4-I_pred"] = cyp3a4i_pred_binary
print(data_add_score)

### CYP2C9-I
# get the cyp2c9-i encode
cyp2c9i_drug_encoding = "GIN_ContextPred"
cyp2c9i_data_encoding_single = data_merge.smiles_encode(data_smiles, cyp2c9i_drug_encoding)
data["drug_encoding"] = cyp2c9i_data_encoding_single
# predict the result and add the pred to df
cyp2c9i_model = CompoundPred.model_pretrained("../best-model/GIN_ContextPred-CYP2C9-I_Cmodel")
cyp2c9i_pred = cyp2c9i_model.predict(data)
cyp2c9i_pred_binary = np.asarray([1 if i else 0 for i in (np.asarray(cyp2c9i_pred) >= 0.5)])
data_add_score["CYP2C9-I_pred"] = cyp2c9i_pred_binary
print(data_add_score)


### CYP2D6-S
# get the cyp2d6-s encode
cyp2d6s_drug_encoding = "GIN_ContextPred"
cyp2d6s_data_encoding_single = data_merge.smiles_encode(data_smiles, cyp2d6s_drug_encoding)
data["drug_encoding"] = cyp2d6s_data_encoding_single
# predict the result and add the pred to df
cyp2d6s_model = CompoundPred.model_pretrained("../best-model/GIN_ContextPred-CYP2D6-S_Cmodel")
cyp2d6s_pred = cyp2d6s_model.predict(data)
cyp2d6s_pred_binary = np.asarray([1 if i else 0 for i in (np.asarray(cyp2d6s_pred) >= 0.5)])
data_add_score["CYP2D6-S_pred"] = cyp2d6s_pred_binary
print(data_add_score)

### CYP3A4-S
# get the cyp3a4-s encode
cyp3a4s_drug_encoding = "CNN"
cyp3a4s_data_encoding_single = data_merge.smiles_encode(data_smiles, cyp3a4s_drug_encoding)
data["drug_encoding"] = cyp3a4s_data_encoding_single
# predict the result and add the pred to df
cyp3a4s_model = CompoundPred.model_pretrained("../best-model/CNN-CYP3A4-S_Cmodel")
cyp3a4s_pred = cyp3a4s_model.predict(data)
cyp3a4s_pred_binary = np.asarray([1 if i else 0 for i in (np.asarray(cyp3a4s_pred) >= 0.5)])
data_add_score["CYP3A4-S_pred"] = cyp3a4s_pred_binary
print(data_add_score)


### CYP2C9-S
# get the cyp2c9-s encode
cyp2c9s_drug_encoding = "GIN_ContextPred"
cyp2c9s_data_encoding_single = data_merge.smiles_encode(data_smiles, cyp2c9s_drug_encoding)
data["drug_encoding"] = cyp2c9s_data_encoding_single
# predict the result and add the pred to df
cyp2c9s_model = CompoundPred.model_pretrained("../best-model/GIN_ContextPred-CYP2C9-S_Cmodel")
cyp2c9s_pred = cyp2c9s_model.predict(data)
cyp2c9s_pred_binary = np.asarray([1 if i else 0 for i in (np.asarray(cyp2c9s_pred) >= 0.5)])
data_add_score["CYP2C9-S_pred"] = cyp2c9s_pred_binary
print(data_add_score)


### CYP2C19-I
cyp2c19i_drug_encoding = "GIN_ContextPred"
cyp2c19i_data_encoding_single = data_merge.smiles_encode(data_smiles, cyp2c19i_drug_encoding)
data["drug_encoding"] = cyp2c19i_data_encoding_single
# predict the result and add the pred to df
cyp2c19i_model = CompoundPred.model_pretrained("../best-model/GIN_ContextPred-CYP2C19-I_Cmodel")
cyp2c19i_pred = cyp2c19i_model.predict(data)
cyp2c19i_pred_binary = np.asarray([1 if i else 0 for i in (np.asarray(cyp2c19i_pred) >= 0.5)])
data_add_score["CYP2C19-I_pred"] = cyp2c19i_pred_binary
print(data_add_score)


### CYP1A2-I
cyp1a2i_drug_encoding = "GIN_ContextPred"
cyp1a2i_data_encoding_single = data_merge.smiles_encode(data_smiles, cyp1a2i_drug_encoding)
data["drug_encoding"] = cyp1a2i_data_encoding_single
# predict the result and add the pred to df
cyp1a2i_model = CompoundPred.model_pretrained("../best-model/GIN_ContextPred-cyp1a2_Cmodel")
cyp1a2i_pred = cyp1a2i_model.predict(data)
cyp1a2i_pred_binary = np.asarray([1 if i else 0 for i in (np.asarray(cyp1a2i_pred) >= 0.5)])
data_add_score["CYP1A2-I_pred"] = cyp1a2i_pred_binary


"""E ：Excretion"""
### Half_Life
# get the encode smiles
hl_drug_encoding = "rdkit_2d_normalized"
hl_test_data = data_merge.smiles_ml_encode(data, data_smiles, hl_drug_encoding)
# predict the result and add the pred to df
with open("../best-model/SVR-Half-life_Rmodel/model.pickle", "rb") as file:
    hl_model = pickle.load(file)
    print(hl_model)

# print(vd_test_data)
hl_pred = hl_model.predict(hl_test_data)
data_add_score["Half-Life_pred"] = hl_pred
print(data_add_score)


### CL-Micro
# get the CL-Micro encode
clm_drug_encoding = "GIN_ContextPred"
clm_data_encoding_single = data_merge.smiles_encode(data_smiles, clm_drug_encoding)
print(clm_data_encoding_single)
data["drug_encoding"] = clm_data_encoding_single
print(data)
# predict the result and add the pred to df
clm_model = CompoundPred.model_pretrained("../best-model/GIN_ContextPred_CL-Micro_Rmodel")
clm_pred = clm_model.predict(data)
data_add_score["CL-Micro_pred"] = clm_pred
print(data_add_score)

### CL-Hepa
# get the CL-Hepa encode
clh_drug_encoding = "GIN_ContextPred"
clh_data_encoding_single = data_merge.smiles_encode(data_smiles, clh_drug_encoding)
data["drug_encoding"] = clh_data_encoding_single
# predict the result and add the pred to df
clh_model = CompoundPred.model_pretrained("../best-model/GIN_ContextPred_CL-Hepa_Rmodel")
clh_pred = clh_model.predict(data)
data_add_score["CL-Hepa_pred"] = clh_pred
print(data_add_score)


"""T：Toxicity"""
### hERG
# get the hERG encode
herg_drug_encoding = "RDKit2D"
herg_data_encoding_single = data_merge.smiles_encode(data_smiles, herg_drug_encoding)
data["drug_encoding"] = herg_data_encoding_single
# predict the result and add the pred to df
herg_model = CompoundPred.model_pretrained("../best-model/RDKit2D-hERG_Cmodel")
herg_pred = herg_model.predict(data)
herg_pred_binary = np.asarray([1 if i else 0 for i in (np.asarray(herg_pred) >= 0.5)])
data_add_score["hERG_pred"] = herg_pred_binary
print(data_add_score)


### AMES
# get the AMES encode
ames_drug_encoding = "GIN_AttrMasking"
ames_data_encoding_single = data_merge.smiles_encode(data_smiles, ames_drug_encoding)
data["drug_encoding"] = ames_data_encoding_single
# predict the result and add the pred to df
ames_model = CompoundPred.model_pretrained("../best-model/GIN_AttrMasking-ames_Cmodel")
ames_pred = ames_model.predict(data)
ames_pred_binary = np.asarray([1 if i else 0 for i in (np.asarray(ames_pred) >= 0.5)])
data_add_score["AMES_pred"] = ames_pred_binary
print(data_add_score)


### DILI
# get the DILI encode
dili_drug_encoding = "GIN_AttrMasking"
dili_data_encoding_single = data_merge.smiles_encode(data_smiles, dili_drug_encoding)
data["drug_encoding"] = dili_data_encoding_single
# predict the result and add the pred to df
dili_model = CompoundPred.model_pretrained("../best-model/GIN_AttrMasking-dili_Cmodel")
dili_pred = dili_model.predict(data)
dili_pred_binary = np.asarray([1 if i else 0 for i in (np.asarray(dili_pred) >= 0.5)])
data_add_score["DILI_pred"] = dili_pred_binary
print(data_add_score)


###
# get the LD50 encode
ld50_drug_encoding = "GIN_AttrMasking"
ld50_data_encoding_single = data_merge.smiles_encode(data_smiles, ld50_drug_encoding)
data["drug_encoding"] = ld50_data_encoding_single
# predict the result and add the pred to df
ld50_model = CompoundPred.model_pretrained("../best-model/GIN_AttrMasking-ld50_Rmodel")
ld50_pred = ld50_model.predict(data)
data_add_score["LD50_pred"] = ld50_pred
print(data_add_score)


### carcinogens
# get the carcinogens encode
carc_drug_encoding = "CNN"
carc_data_encoding_single = data_merge.smiles_encode(data_smiles, carc_drug_encoding)
data["drug_encoding"] = carc_data_encoding_single
# predict the result and add the pred to df
carc_model = CompoundPred.model_pretrained("../best-model/CNN-carcinogens_Cmodel")
carc_pred = carc_model.predict(data)
carc_pred_binary = np.asarray([1 if i else 0 for i in (np.asarray(carc_pred) >= 0.5)])
data_add_score["Carcinogens_pred"] = carc_pred_binary
print(data_add_score)

### clintox
# get the clintox encode
clintox_drug_encoding = "GIN_AttrMasking"
clintox_data_encoding_single = data_merge.smiles_encode(data_smiles, clintox_drug_encoding)
data["drug_encoding"] = clintox_data_encoding_single
# predict the result and add the pred to df
clintox_model = CompoundPred.model_pretrained("../best-model/AttentiveFP-clintox_Cmodel-500")
clintox_pred = clintox_model.predict(data)
clintox_pred_binary = np.asarray([1 if i else 0 for i in (np.asarray(clintox_pred) >= 0.5)])
data_add_score["ClinTox_pred"] = clintox_pred_binary
print(data_add_score)


### skin-reaction
# get the skin-reaction encode
skin_drug_encoding = "GIN_AttrMasking"
skin_data_encoding_single = data_merge.smiles_encode(data_smiles, skin_drug_encoding)
data["drug_encoding"] = skin_data_encoding_single
# predict the result and add the pred to df
skin_model = CompoundPred.model_pretrained("../best-model/GIN_ContextPred-skin_reaction_Cmodel")
skin_pred = skin_model.predict(data)
skin_pred_binary = np.asarray([1 if i else 0 for i in (np.asarray(skin_pred) >= 0.5)])
data_add_score["Skin_Reaction_pred"] = skin_pred_binary
print(data_add_score)



###  save the csv to file

data.to_csv(args.admet_result_path, mode="a", index=False)
"""---------------------------------------------------------"""
