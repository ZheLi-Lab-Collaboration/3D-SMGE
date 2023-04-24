# -*- coding: utf-8 -*-
# @Time    : 2022/4/22 17:02
# @Author  : TkiChus
# @Email   : XU.Chao.TkiChus@gmail.com

from guacamol.utils.chemistry import canonicalize_list, is_valid, calculate_pc_descriptors, continuous_kldiv, \
    discrete_kldiv, calculate_internal_pairwise_similarities

from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit import rdBase, Chem
from rdkit.Chem import AllChem, Descriptors

import torch
import pandas as pd
from SA_Score import sa_score

"""example"""
# mol = Chem.MolFromSmiles('O=C1N(CC2=CC=C(F)C=C2)C(N[C@H](C)C3=CC=C(Cl)C=C3)=NC4=C1C=NN4C(C)(C)C')
# path = "../data/pde1.mol"
# mol = Chem.MolFromMolFile(path)
# print(Chem.MolToSmiles(m))
# a = sa_score(mol)
# print(a)


# mol = Chem.MolToSmiles("C[C@@H](Nc1nc2c(cnn2C(C)(C)C)c(=O)n1Cc1ccc(F)cc1)c1ccc(Cl)cc1")
# descriptor_names = [x[0] for x in Descriptors._descList]
# des_list = ['MolLogP', 'qed', 'TPSA', 'NumHAcceptors', 'NumHDonors',  'NumRotatableBonds', 'NumAliphaticRings']
# calculator = MoleculeDescriptors.MolecularDescriptorCalculator(des_list)
# list = calculator.CalcDescriptors(mol)
# print(list)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', type=str, help='Path to directory containing csv_files to put into ADMET prediction.'
                                                 'The generated molecules of SMILES format are converted into csv format',
                    default='../data/csv_files/')
parser.add_argument('--baseP_result_path', type=str, help='Path to directory containing the result of final ADMET prediction.',
                    default='../data/csv_files/')

args = parser.parse_args()


def cal_baseFeature(input_smiles_path, final_result_path):
    """
    Args:
        input_smiles_path: generate smiles csv file path
        final_result_path: final file path to save result

    Returns:
        a csv to save the final result
    """
    data = pd.read_csv(input_smiles_path)
    smiles_data = data["SMILES"]
    # print(smiles_data.shape[0])
    # print(smiles_data[0])
    # print(smiles_data)
    list_sa = []
    list_logP = []
    list_qed = []
    list_TPSA = []
    list_NHA = []
    list_NHD = []
    list_NRB = []
    list_NAR = []

    for i in range(smiles_data.shape[0]):
        mol = Chem.MolFromSmiles(smiles_data[i])
        des_list = ['MolLogP', 'qed', 'TPSA', 'NumHAcceptors', 'NumHDonors', 'NumRotatableBonds', 'NumAliphaticRings']
        calculator = MoleculeDescriptors.MolecularDescriptorCalculator(des_list)
        list_single = calculator.CalcDescriptors(mol)
        sa_scores = sa_score(mol)
        # print(list_single)
        list_logP.append(list_single[0])
        list_qed.append(list_single[1])
        list_TPSA.append(list_single[2])
        list_NHA.append(list_single[3])
        list_NHD.append(list_single[4])
        list_NRB.append(list_single[5])
        list_NAR.append(list_single[6])
        list_sa.append(sa_scores)
    data["MolLogP"] = list_logP
    data["QED"] = list_qed
    data["SA Score"] = list_sa
    data["TPSA"] = list_TPSA
    data["NumHAcceptors"] = list_NHA
    data["NumHDonors"] = list_NHD
    data["NumRotatableBonds"] = list_NRB
    data["NumAliphaticRings"] = list_NAR

    data.to_csv(final_result_path, mode="a", index=False)
    print("----achieved!!!")


if __name__ == '__main__':
    cal_baseFeature(args.csv_path, args.baseP_result_path)