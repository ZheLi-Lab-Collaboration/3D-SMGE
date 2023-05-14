#!/bin/bash
# load CUDNN
module load cuda/10.2
module load cudnn/7.6.5.32_cuda10.2
module load nccl/2.9.6-1_cuda10.2
module load gcc/9.3

# load anaconda
module load anaconda/2020.11
source activate 3d
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# model train 
torchrun --standalone --nnodes=1 --nproc_per_node=4 SMGE_3D_parallel.py train 3D_SMG ./data/ ./model --split 37905 2527 --cuda --parallel --batch_size 5 --draw_random_samples 5 --features 128 --interactions 7 --caFilter_per_block 4 --max_epochs 1000

# model eval 
torchrun --standalone --nnodes=1 --nproc_per_node=1 SMGE_3D_eval_single_gpu.py eval 3D_SMG  ./data/ ./model --split validation --cuda --parallel --batch_size 5 --features 128 --interactions 7 --caFilter_per_block 4

# model generate

torchrun --standalone --nnodes=1 --nproc_per_node=1 SMGE_3D_eval_single_gpu.py generate 3D_SMG ./model/ 200 --scaffold 'CC(C1=CC=C(OC)C(OC)=C1)=O' --chunk_size 2000 --cuda --max_length 65 --file_name scaffold


# filter 
python filter_generated.py ./model/generated/scaffold.mol_dict  


# ASE DB -> .xyz

python write_xyz_files.py ./model/generated/

# .xyz -> .smi
python xyz_to_smiles.py ./model/generated/


