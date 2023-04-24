# -*- coding: utf-8 -*-
# @Time    : 2022/3/30 21:15
# @Author  : TkiChus
# @Email   : XU.Chao.TkiChus@gmail.com

from openbabel.pybel import readfile, Outputfile
import os
import re
import pandas as pd
import sys


def MolFormatConversion(path, input_format="xyz", output_format="smi"):
    
    input_path = os.path.join(path, "generated_xyz_files")
    if not os.path.exists(input_path):
        os.mkdir(input_path)
    output_path = os.path.join(path, "generated_smi")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    ordered_files = sorted(os.listdir(input_path), key=lambda x: re.sub('.xyz', '', x))

    total = 0
    for i, xyzfile in enumerate(ordered_files):
        xyzfile = os.path.join(input_path, xyzfile)
        print(xyzfile)

        output_tail = str(i)+".smi"
        output_file_smi = os.path.join(output_path, output_tail)

        print(output_file_smi)
        molecules = readfile(input_format, xyzfile)
        output_file_writer = Outputfile(output_format, output_file_smi)

        for i, molecule in enumerate(molecules):
            output_file_writer.write(molecule)
        output_file_writer.close()
        total += 1
    print('%d molecules converted'%(total+1))


def smi_aggOne(merge_path):
    origin_path = os.path.join(path, "generated_smi")
    filenames = os.listdir(origin_path)
    merge_path = os.path.join(merge_path, "agg_smi")
    if not os.path.exists(merge_path):
        os.mkdir(merge_path)
    merge_path = merge_path + "/agg_smi.smi"
    file = open(merge_path, 'w', encoding='utf8')
    for filename in filenames:
        filepath = os.path.join(origin_path, filename)
        for line in open(filepath, encoding='utf8'):
            smi_line = line.split("..")[0]
            file.writelines(smi_line)
        file.write('\n')
    file.close()



if __name__ == '__main__':
    """test xyy to smi"""
    path = sys.argv[1]
    MolFormatConversion(path)
    """merge txt"""
    smi_aggOne(path)








