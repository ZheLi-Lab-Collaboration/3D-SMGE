# -*- coding: utf-8 -*-
# @Time    : 2022/6/10 20:15
# @Author  : TkiChus
# @Email   : XU.Chao.TkiChus@gmail.com


import numpy as np
# from molecule import pybel, GetSymbol

try:
    from openbabel import pybel  # openbabel 3.1.1
    GetSymbol = pybel.ob.GetSymbol
    GetVdwRad = pybel.ob.GetVdwRad
except ImportError:
    import pybel  # openbabel 2.4
    table = pybel.ob.OBElementTable()
    GetSymbol = table.GetSymbol
    GetVdwRad = table.GetVdwRad

def generate_conformations_from_openbabel(smiles, num_conf, ob_gen3D_option='best'):
    # initialize obmol
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


    return atomic_numbers_fg, conformer_coordinates

def extract_from_obmol(mol, ) -> tuple([list, np.array, np.ndarray, np.ndarray]):
    """Extract information from Openbabel OBMol object with conformers."""
    atom_atomic_number_dict = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "Cl": 17, 'S': 16}

    py_mol = pybel.Molecule(mol)
    py_mol.localopt(forcefield='mmff94', steps=50)
    elements = [GetSymbol(atom.atomicnum) for atom in py_mol.atoms]
    charges = np.array([atom.formalcharge for atom in py_mol.atoms])

    n_atoms = len(py_mol.atoms)
    connectivity_matrix = np.zeros((n_atoms, n_atoms))
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
        mol.SetConformer(i)
        coordinates = list([atom.coords for atom in py_mol.atoms])
        conformer_coordinates.append(coordinates)
        for i in range(len(elements)):
            atomic_numbers_fg.append(atom_atomic_number_dict[elements[i]])
    # conformer_coordinates = conformer_coordinates

    return atomic_numbers_fg, conformer_coordinates[0], connectivity_matrix, charges



if __name__ == '__main__':
    a, b = generate_conformations_from_openbabel("O=C1[N]C(=S)N=C2N=CC=N[C]12", 50)
    print(a, b)