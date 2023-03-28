import os
import numpy as np
import math as m
from pymatgen.core import structure


class UCF:

    def __init__(self, atom_type, atom_num, exchange_energy, atom_coordinate, mat_lc_hc, unit_cell_size, move_distence,
                 interaction_type, dimension=2):
        self.atom_type = atom_type
        self.atom_num = atom_num
        self.exchange_Energy = exchange_energy
        self.Dimension = dimension
        self.Unit_Cell_Size = unit_cell_size
        self.Atom_coordinate = atom_coordinate
        self.Mat_Lc_Hc = mat_lc_hc
        self.move_distence = move_distence
        self.interaction_type = interaction_type

    def Get_exchange_energy(self):
        return self.exchange_Energy

    def Get_dimension(self):
        return self.Dimension

    def Get_atom_coordiante(self):
        return self.Atom_coordinate

    def Get_Mat_Lc_Hc(self):
        return self.Mat_Lc_Hc

    def mkdir(self, Path):
        folder = os.path.exists(Path)
        if not folder:
            os.makedirs(Path)
            return Path
        else:
            return Path

    def interaction_num(self):
        lenNN = len(UCF.Create_interanction_NN(self))
        lenNNN = len(UCF.Creat_interaction_NNN(self))
        lenNNNN = len(UCF.Creat_interaction_NNNN(self))
        return lenNN, lenNNN, lenNNNN, lenNN + lenNNN + lenNNNN

    def Create_Dimension(self):
        if UCF.Get_dimension(self) == 3:
            diagonal = np.array([1.0, 1.0, 1.0])
            vectors = np.diag(diagonal)
        else:
            diagonal = np.array([1.0, 1.0, 0.0])
            vectors = np.diag(diagonal)
        return vectors.tolist()

    def Create_atom_num(self):
        k = [[] for i in range(self.atom_num)]
        for i in range(len(UCF.Get_atom_coordiante(self)[1])):
            k[i].append(i)
            k[i].append(UCF.Get_atom_coordiante(self)[1][i])
            k[i].append(UCF.Get_Mat_Lc_Hc(self)[i])
        return k

    def move(self, a, b, c, d):
        for i in range(len(b)):
            k = []
            k.append((np.array(a[0]) + b[i]).tolist())
            k.append([])
            for j in a[1]:
                k[1].append((np.array(j) + d * b[i]).tolist())
            c[i] = k
        return c

    def inital(self, atom_coordinate):
        origin = atom_coordinate
        move_distence = self.move_distence
        if UCF.Get_dimension(self) == 3:
            move_tensor_3d = np.array(
                [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0], [-1, 1, 0],
                 [0, 0, 0],
                 [1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1], [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
                 [0, 0, 1],
                 [1, 0, -1], [0, 1, -1], [-1, 0, -1], [0, -1, -1], [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
                 [0, 0, -1]])
            coordinate = [[] for i in range(len(move_tensor_3d))]
            p = UCF.move(self, origin, move_tensor_3d, coordinate, move_distence)
        else:
            move_tensor_2d = np.array(
                [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0], [-1, 1, 0],
                 [0, 0, 0]])
            coordinate = [[] for i in range(len(move_tensor_2d))]
            p = UCF.move(self, origin, move_tensor_2d, coordinate, move_distence)
        return p

    def distence(self, a, b):
        d = m.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)
        return round(d, 1)

    def Create_interaction(self):
        NN1 = []
        Distence = set()
        p = UCF.inital(self, UCF.Get_atom_coordiante(self))
        origin = UCF.Get_atom_coordiante(self)
        for i in range(len(p)):
            for num1 in range(len(origin[1])):
                for num2 in range(len(p[i][1])):
                    l = UCF.distence(self, p[i][1][num2], origin[1][num1])
                    if l != 0:
                        NN1.append([num1, num2, p[i][0], l])
                        Distence.add(l)
        q = sorted(Distence)
        return NN1, q

    def Create_interanction_NN(self):
        NN = []
        NN1, q = UCF.Create_interaction(self)
        J = UCF.Get_exchange_energy(self)
        for o in range(len(NN1)):
            if NN1[o][3] <= min(q) + 0.01:
                NN1[o][3] = J[0]
                NN.append(NN1[o])
        return NN

    def Creat_interaction_NNN(self):
        NNN = []
        NN1, q = UCF.Create_interaction(self)
        J = UCF.Get_exchange_energy(self)
        for o in range(len(NN1)):
            if NN1[o][3] > min(q) + 0.01 and NN1[o][3] <= q[1] + 0.01:
                NN1[o][3] = J[1]
                NNN.append(NN1[o])
        return NNN

    def Creat_interaction_NNNN(self):
        NNNN = []
        NN1, q = UCF.Create_interaction(self)
        J = UCF.Get_exchange_energy(self)
        for o in range(len(NN1)):
            if NN1[o][3] > q[1] + 0.01 and NN1[o][3] <= q[2] + 0.01:
                NN1[o][3] = J[2]
                NNNN.append(NN1[o])
        return NNNN

    def write_unit_cell_size(self, path, ucffilename):
        l = self.Unit_Cell_Size
        with open(path + '/' + ucffilename, 'a+') as f:
            f.writelines('# Unit cell size:' + '\n')
            for i in l:
                f.write(str(i) + '\t')
            f.write('\n')
        print('unit_cell_size write success')
        return True

    def write_Unit_cell_vectors(self, path, ucffilename):
        l = UCF.Create_Dimension(self)
        with open(path + '/' + ucffilename, 'a+') as f:
            f.writelines('# Unit cell vectors:' + '\n')
            for i in range(len(l)):
                for j in l[i]:
                    f.write(str(j) + '\t')
                f.write('\n')
        print('Unit_cell_vectors success')
        return True

    def write_Atoms_num(self, path, ucffilename):
        l = UCF.Create_atom_num(self)
        with open(path + '/' + ucffilename, 'a+') as f:
            f.writelines('# Atoms num, id cx cy cz mat lc hc' + '\n')
            f.write(str(self.atom_num) + '\t')
            f.write(str(self.atom_type) + '\n')
            for i in l:
                k = str(i).replace('[', '').replace(']', '').split(',')
                for o in k:
                    f.write(o + '\t')
                f.write('\n')
        print('Atoms num write success')
        return True

    def write_interaction(self, path, ucffilename, data):
        with open(path + '/' + ucffilename, 'a+') as f:
            for i in range(len(data)):
                k = str(data[i]).replace('[', '').replace(']', '').replace("'", '').split(',')
                f.write(str(i) + '\t')
                for o in k:
                    f.write(o + '\t')
                f.write('\n')
        print('interantion write success')
        return True

    def write(self, path, ucffilename):
        UCF.write_unit_cell_size(self, path, ucffilename)
        UCF.write_Unit_cell_vectors(self, path, ucffilename)
        UCF.write_Atoms_num(self, path, ucffilename)
        NN = UCF.Create_interanction_NN(self)
        NNN = UCF.Creat_interaction_NNN(self)
        NNNN = UCF.Creat_interaction_NNNN(self)
        lenNN, lenNNN, lenNNNN, genlen = UCF.interaction_num(self)
        interaction_sum = NN + NNN + NNNN
        with open(path + '/' + ucffilename, 'a+') as f:
            f.writelines('#Interactions n exctype, id i j dx dy   dz        Jij' + '\n')
            f.write(str(genlen) + '\t')
            f.write(self.interaction_type + '\n')
        UCF.write_interaction(self, path, ucffilename, interaction_sum)
        print('all finsh')
        return True


class pymatgen_structure:

    def __init__(self, strut):
        self.strut = strut

    def get_coord(self, struct):
        coord = []
        for i in struct:
            coord.append(i.frac_coords.tolist())
        initial_coord = []
        initial_coord.append([0., 0., 0.])
        initial_coord.append(coord)
        return initial_coord

    def get_unit_cell_size(self, struct):
        a = [i for i in range(3)]
        a[0], a[1], a[2] = struct.lattice.abc
        return a

    def get_atom_sum(self, struct):
        return len(struct)


def main():
    material_name = "Mn2Au"
    cif_path = '/Users/xbunax/Downloads/Mn2Au-2.cif'
    Mn2Au = structure.Structure.from_file(cif_path)
    inital = pymatgen_structure(Mn2Au)
    # atom_coordinate = [[0, 0, 0],
    #                    [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0]]]  # 原子坐标第一个[0,0,0]为位置判断不用改
    atom_coordinate = inital.get_coord(Mn2Au)
    mat_lc_hc = [[1, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]  # 原子材料类型
    exchangeEnergy = ['-10.88E-21','-14.62E-21', '4.18E-21']
    unit_cell_size = inital.get_unit_cell_size(Mn2Au)
    Dimension = 3  # 维度
    move_distence = np.array([1, 1, 0])  # 原包移动向量（x,y,0）
    path = '/Users/xbunax/Downloads/'
    ucffilename = 'Mn2Au3d.ucf'
    atom_num = inital.get_atom_sum(Mn2Au)  # 原包原子个数
    print(atom_num)
    atom_type = 2  # 原包原子种类
    interaction_type = 'isotropic'  # 各向异性和各项同性
    Q = UCF(atom_type, atom_num, exchangeEnergy, atom_coordinate, mat_lc_hc, unit_cell_size, move_distence,
            interaction_type, Dimension)
    Q.write(path, ucffilename)


if __name__ == "__main__":
    main()
