from ase.db import connect
import os
import re
from ase.io.extxyz import read_xyz
from ase.units import Debye, Bohr, Hartree, eV
from schnetpack import Properties
import shutil
import tempfile
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles as MSS
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--xyz_path', type=str, help='Path to directory containing xyz_files for creating dataset for train deep generative model',
                    default='./xyz_files/')

args = parser.parse_args()


# raw_path = './ZINC_7w_noS6/'
invalid_path = "./fun_grp_invalid.txt"
tmpdir = tempfile.mkdtemp('temporary')

with connect(os.path.join('SMGE3D.db')) as con:
    # ordered_files = sorted(os.listdir(raw_path),
    #                        key=lambda x: (int(re.sub('\D', '', x)), x))
    #                        key=lambda x: re.sub('.xyz', '', x))
    ordered_files = sorted(os.listdir(args.xyz_path),
                           key=lambda x: int(x.split(".")[0].split("_")[1]))
    # print(ordered_files)
    print("count:", len(ordered_files))
    for i, xyzfile in enumerate(ordered_files):
        print(xyzfile)
        xyzfile = os.path.join(args.xyz_path, xyzfile)
        if (i + 1) % 100 == 0:
            print('Parsed: {:6d} / 50000'.format(i + 1))
        properties = {}
        invalid_ids = []
        tmp = os.path.join(tmpdir, 'tmp.xyz')
        smiles_string_func_grp = ''
        with open(xyzfile, 'r') as f:
            lines = f.readlines()
            # print("file:", xyzfile)
            print(i)
            if (len(lines) > 0):
                l = lines[1].split()[2:]
                #smiles_string_molecule = lines[-2].strip().split("\t")[0]
                #smiles_string_func_grp = lines[-1].strip().split("\t")[0]

                # smiles_string_molecule = lines[-1].strip().split("\t")[0]
                smiles_string_molecule = lines[-1].strip()
                #smiles_string_func_grp = lines[-2].strip().split("\t")[0]
                smiles_string_func_grp = MSS(smiles_string_molecule)
                if (smiles_string_func_grp != ''):
                    print("SMILES:", smiles_string_molecule)
                    print("SMILES function", smiles_string_func_grp)
                    properties['Smiles_String'] = smiles_string_molecule
                    properties['Functional_Group'] = smiles_string_func_grp
                    # print("properties----", properties)
                    with open(tmp, "wt") as fout:
                        for line in lines:
                            fout.write(line.replace('*^', 'e'))
                            print("line.replace('*^', 'e')----", line.replace('*^', 'e'))


                    with open(tmp, 'r') as f:
                        ats = list(read_xyz(f, 0))[0]  # Atoms
                    # print(properties)
                    con.write(ats, data=properties)

                else:
                    # if you prepare data firstly, please run it
                    invalid_ids.append(i+1)
                    with open(invalid_path, "a", encoding='utf-8') as w:
                        w.write(str(i+1))
                        w.write("\n")

            else:
                continue

        # if (smiles_string_func_grp != ''):
        #     with open(tmp, 'r') as f:
        #         ats = list(read_xyz(f, 0))[0]#Atoms
        #
        #     print("properties_xyz---",  properties)
        #     con.write(ats, data=properties)

print('Done.')
shutil.rmtree(tmpdir) # Delete a complete directory tree

os.system('python {} {}'.format('preprocess_dataset.py', './SMG3D.db'))











