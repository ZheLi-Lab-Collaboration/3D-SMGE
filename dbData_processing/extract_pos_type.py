# -*- coding: utf-8 -*-
# @Time    : 2022/12/14 14:47
# @Author  : TkiChus
# @Email   : XU.Chao.TkiChus@gmail.com



"""if you want using the existing 3D datasets, like SDF of ZINC 3D.
    You just extract the pos and type of SDF.
    We create an example of processing a ZINC SDF file, other types of files are similar,
    and fine-tuning the script will do the trick!"""
import os


def extract_pos_type_sdf(sdf_path, xyz_path, start_line):
    """

    Args:
        sdf_path: the path of file
        start_line: start from the line，you can change it according to the format

    Returns:

    """
    atom_atomic = ["H", "C", "N", "O", "F", "Cl", "S"]
    every_line_write = []
    total_line_write = []
    total_atom = 0


    with open(sdf_path) as f:
        lines = f.readlines()

        for line in range(start_line, len(lines)):
            if ((lines[line].strip()).split())[3] in atom_atomic:

                every_line_write.append(((lines[line].strip()).split())[3])
                every_line_write.append(((lines[line].strip()).split())[0])
                every_line_write.append(((lines[line].strip()).split())[1])
                every_line_write.append(((lines[line].strip()).split())[2])

                total_line_write.append(every_line_write)
                every_line_write = []
                total_atom += 1

            else:
                break
        print(total_line_write)


    with open(xyz_path, "a+", encoding="utf8") as w:
        if not os.path.exists(xyz_path):
            os.mkdir(xyz_path)

        w.write(str(total_atom))
        w.write("\n\n")
        for every_pos_type in total_line_write:
            for i in every_pos_type:
                w.write(i.ljust(10))

            w.write("\n")


    f.close()
    w.close()

if __name__ == '__main__':

    extract_pos_type_sdf("./test.sdf", "./test.xyz", 4)