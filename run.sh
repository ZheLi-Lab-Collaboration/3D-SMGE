#!/bin/bash
#load cuda cudnn
module load cuda/10.2
module load cudnn/7.6.5.32_cuda10.2
module load gcc/9.3

# load anaconda
module load anaconda/2020.11
source activate 3d
# export CUDA_VISIBLE_DEVICES=0,1,2,3


### model train
# python SMG_3D.py train 3D_SMG ./data/ ./model --split 780 260 --cuda --batch_size 5 --draw_random_samples 5 --features 128 --interactions 7 --caFilter_per_block 4 --max_epochs 1000


### model eval
# python SMG_3D.py eval 3D_SMG  ./data/ ./model --split validation --cuda --batch_size 5 --features 128 --interactions 7 --caFilter_per_block 4


### model generate
### new generate mode
# mode1 smiles
# python SMG_3D.py generate 3D_SMG ./model/ 100 --scaffold 'CC(C1=CC=C(OC)C(OC)=C1)=O' --genMode mode1 --inputFormat smiles --chunk_size 100 --cuda --max_length 60 --file_name scaffold
# mode1 pdb
# python SMG_3D.py generate 3D_SMG ./model/ 100 --genMode mode1 --inputFormat pdb --file3D_path ./pdb_luo.pdb --chunk_size 100 --cuda --max_length 60 --file_name scaffold
# mode1 mol2
# python SMG_3D.py generate 3D_SMG ./model/ 100 --genMode mode1 --inputFormat mol2 --file3D_path ./pdb_luo.mol2 --chunk_size 100 --cuda --max_length 60 --file_name scaffold

# mode2 smiles
# python SMG_3D.py generate 3D_SMG ./model/ 100 --scaffold 'CC(C1=CC=C(OC)C(OC)=C1)=O' --genMode mode2 --have_finished  1 2 3 4 6 7 8 9 10 11 12 13 --inputFormat smiles --chunk_size 100 --cuda --max_length 60 --file_name scaffold
# mode2 pdb
# python SMG_3D.py generate 3D_SMG ./model/ 100 --genMode mode2 --inputFormat pdb --file3D_path ./pdb_luo.pdb --chunk_size 100 --cuda --max_length 60 --file_name scaffold
# mode2 mol2
# python SMG_3D.py generate 3D_SMG ./model/ 100 --genMode mode2 --inputFormat mol2 --file3D_path ./pdb_luo.mol2 --chunk_size 100 --cuda --max_length 60 --file_name scaffold

#---------------------------------------------------------------------------------------------




# model filter
# python filter_generated.py ./model/generated/scaffold.mol_dict 


# ASE DB -> .xyz
# python write_xyz_files.py ./model/generated/


# .xyz -> .smi
# python xyz_to_smiles.py ./model/generated



