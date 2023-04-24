# -*- coding: utf-8 -*-
# @Time    : 2022/3/31 21:28
# @Author  : TkiChus
# @Email   : XU.Chao.TkiChus@gmail.com

import rdkit
from rdkit import Chem
from rdkit.Chem import Draw



"""

# Implement a procedure to assess the valence of generated molecules here!
# You can adapt and use the Molecule class in utility_classes.py,
# but the current code is tailored towards the QM9 dataset. In fact,
# the OpenBabel algorithm to kekulize bond orders is not very reliable
# and we implemented some heuristics in the Molecule class to fix these
# flaws for structures made of C, N, O, and F atoms. However, when using
# more complex structures with a more diverse set of atom types, we think
# that the reliability of bond assignment in OpenBabel might further
# degrade and therefore do no recommend to use valence checks for
# analysis unless it is very important for your use case.
"""

def check_vality(smiles):
    total = 0
    with open(smiles, "r") as f:
        a = f.readlines()
        for i in range(len(a)):
            smiles_ = a[i]
            try:
                mol = Chem.MolFromSmiles(smiles_)
                vai_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                print(vai_smiles)
                total += 1
            except Exception as e:
                pass
            continue
    print("all valid molecules %d" % (total))
    val_ratio = float(total / len(a))
    print("valid ratio %.2f" % (val_ratio))


# all molecules(no duplicate) / all valid generate molecules
def check_uniqueness(smiles):

    ### 1057
    with open(smiles, "r") as f:
        lines = f.readlines()
        print(lines)
        smiles_fg = set(lines)
        print(len(smiles_fg))
        for smiles in smiles_fg:
            with open("final_un.smi", "a+", encoding="utf8") as a:
                a.writelines(smiles)
        un_ratio = float(len(smiles_fg) / len(lines))
        print("valid ratio %.2f" % (un_ratio))
    f.close()



#  all novel molecules / all datasets molecules
def check_novelty(train_smi, genUn_smi):
    count = 0
    # 1057
    with open(train_smi, "r") as tr:
        train = tr.readlines()
        print(train)
        with open(genUn_smi, "r") as gr:
            gen = gr.readlines()
            for line in gen:
                if line not in train:
                    count += 1

        print(count)
    novelty_ratio = float(len(gr.readlines()) / len(train))
    print("novelty ratio %.2f" % (novelty_ratio))
    tr.close()
    gr.close()



