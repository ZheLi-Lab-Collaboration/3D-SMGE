# -*- coding: utf-8 -*-
# @Time    : 2022/8/24 8:51
# @Author  : TkiChus
# @Email   : XU.Chao.TkiChus@gmail.com

import numpy as np
from rdkit import Chem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import AllChem
import rdkit.DataStructs as DataStructs

from openbabel import openbabel
import ase.io
from ase import Atoms
from ase.constraints import FixAtoms
import sys
import eventlet

import func_timeout

log_print = open('Defalust.log', 'a+')
sys.stdout = log_print
sys.stderr = log_print

""""
Provide a variety of data set input formats to give users more choices
"""

"""openbabel"""
try:
    from openbabel import pybel  # openbabel 3.1.1
    GetSymbol = pybel.ob.GetSymbol
    GetVdwRad = pybel.ob.GetVdwRad
except ImportError:
    import pybel  # openbabel 2.4
    table = pybel.ob.OBElementTable()
    GetSymbol = table.GetSymbol
    GetVdwRad = table.GetVdwRad

def smiles_to_xyz(smiles, num_conf=3, ob_gen3D_option='best'):


    obmol = pybel.readstring('smi', smiles).OBMol

    # initial geometry
    gen3D = pybel.ob.OBOp.FindType("gen3D")
    gen3D.Do(obmol, ob_gen3D_option)

    # conf search
    confSearch = pybel.ob.OBConformerSearch()
    confSearch.Setup(obmol, num_conf)
    confSearch.Search()
    confSearch.GetConformers(obmol)

    atomic_numbers_fg, conformer_coordinates, connectivity_matrix, charges = extract_from_obmol(obmol)
    # conformer_coordinates = list(conformer_coordinates)
    print(atomic_numbers_fg)
    print(conformer_coordinates)
    # print(conformer_coordinates[0][2])
    # print(type(conformer_coordinates))
    print("-----------------------------------------------------------")


    return atomic_numbers_fg, conformer_coordinates

def extract_from_obmol(mol, ) -> tuple([list, np.array, np.ndarray, np.ndarray]):
    """Extract information from Openbabel OBMol object with conformers."""
    atom_atomic_number_dict = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "Cl": 17, 'S': 16}
    py_mol = pybel.Molecule(mol)
    # mol.DeleteHydrogens()
    elements = [GetSymbol(atom.atomicnum) for atom in py_mol.atoms]
    charges = np.array([atom.formalcharge for atom in py_mol.atoms])

    n_atoms = len(py_mol.atoms)
    connectivity_matrix = np.zeros((n_atoms, n_atoms))
    obConversion = openbabel.OBConversion()
    for bond in pybel.ob.OBMolBondIter(mol):
        i = bond.GetBeginAtomIdx() - 1
        j = bond.GetEndAtomIdx() - 1
        bo = bond.GetBondOrder()
        connectivity_matrix[i, j] = bo
        connectivity_matrix[j, i] = bo

    # Retrieve conformer coordinates
    conformer_coordinates = []
    atomic_numbers_fg = []


    for i in range(mol.NumConformers()):

        mol.AddHydrogens()
        # mol.SetConformer(i)

        obConversion.WriteFile(mol, 'change.xyz')
        coordinates = list([atom.coords for atom in py_mol.atoms])
        conformer_coordinates.append(np.array(coordinates))

        for i in range(len(elements)):
            atomic_numbers_fg.append(atom_atomic_number_dict[elements[i]])

        geo = Atoms()
        for atom in py_mol:
            atom_type = atom.atomicnum
            atom_position = np.array([float(i) for i in np.array(atom.coords)])

            geo.append(atom_type)
            geo.positions[-1] = atom_position

        ase.io.write('_ase.xyz', geo, format='xyz', fmt="%18.10f")



    return atomic_numbers_fg, conformer_coordinates[0], connectivity_matrix, charges




def smilesSimilarity(origin_path, convert_path, success_path):
    file_success = open(success_path, "a", encoding="utf8")
    file_convert = open(convert_path, "r", encoding="utf8")
    with open(origin_path, "r", encoding="utf8") as of:
        for oline in of.readlines():
            for cline in file_convert.readlines():
                omlo = Chem.MolFromSmiles(oline)
                cmol = Chem.MolFromSmiles(cline)
                similarity = DataStructs.FingerprintSimilarity(FingerprintMols.FingerprintMol(omlo), FingerprintMols.FingerprintMol(cmol))
                if similarity > 0.95:
                    file_success.writelines(oline)


    file_convert.close()
    file_success.close()
    of.close()

import signal
import time

def time_limit(num, callback):
    def wrap(func):
        def handle(signum, frame):
            raise RuntimeError
        def to_do(*args, **kwargs):
            try:
                signal.signal(signal.SIGALRM, handle)
                signal.alarm(num)
                print('start alarm signal.')
                r = func(*args, **kwargs)
                print('close alarm signal.')
                signal.alarm(0)
                return r
            except RuntimeError as e:
                callback()
        return to_do
    return wrap

def after_timeout():
    print("time out")

@func_timeout.func_set_timeout(5)
def babel_xyz(inter_j, smiles, mol, num_conf=1, ob_gen3D_option='best'):
    conformer_coordinates = []



    gen3D = pybel.ob.OBOp.FindType("gen3D")
    gen3D.Do(mol, ob_gen3D_option)

    # conf search
    confSearch = pybel.ob.OBConformerSearch()
    confSearch.Setup(mol, num_conf)
    confSearch.Search()
    confSearch.GetConformers(mol)
    # atom_atomic_number_dict = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "Cl": 17, 'S': 16}
    py_mol = pybel.Molecule(mol)
    # forcefield optomization
    py_mol.localopt(forcefield='mmff94', steps=50)

    for i in range(mol.NumConformers()):

        mol.AddHydrogens()
        coordinates = list([np.array(atom.coords) for atom in py_mol.atoms])
        conformer_coordinates.append(np.array(coordinates))
        if np.array(conformer_coordinates).all() == 0:
            file = open("error_index.txt", "a+", encoding="utf8")
            file.writelines(str(inter_j) + "\n")
        else:

            geo = Atoms()
            for atom in py_mol:
                atom_type = atom.atomicnum
                atom_position = np.array([float(i) for i in atom.coords])

                geo.append(atom_type)
                geo.positions[-1] = atom_position
                # print(geo.positions[-1])

            # geo.center()
            c = FixAtoms(indices=[atom.index for atom in geo if atom.symbol == 'XX'])
            geo.set_constraint(c)
            ase.io.write('ZINC_' + str(inter_j) + '.xyz', geo, format='xyz', fmt="%18.10f", parallel=True)
            with open('ZINC_' + str(inter_j) + '.xyz', "a+", encoding="utf8") as smiles_w:
                smiles_w.writelines(smiles)
    print("---------------This is {} smiles-----------".format(inter_j))
    print(conformer_coordinates)
    return conformer_coordinates



def batch_Smiles2xyz(smi_path):
    # initialize obmol

    origin_suppl = Chem.SmilesMolSupplier(smi_path, nameColumn=0)
    origin_smi = [Chem.MolToSmiles(mol) for mol in origin_suppl]

    for j in range(len(origin_smi)):
        mol = pybel.readstring('smi', origin_smi[j]).OBMol
        try:
            conformer_coordinates = babel_xyz(j, origin_smi[j], mol)
        except func_timeout.exceptions.FunctionTimedOut:
            continue

        if conformer_coordinates is None:
        # else:
            file = open("error_index.txt", "a+", encoding="utf8")
            file.writelines("distance geometry or 0:"+str(j) + "\n")



if __name__ == '__main__':
    smiles = "smi path"
    batch_Smiles2xyz(smiles)
