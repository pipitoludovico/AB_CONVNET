# import numpy as np
#
#
# _ = np.zeros(shape=(6, 34))
# _a = np.zeros(shape=(6, 34))
# for zio in ('rec', 'lig'):
#     for i in range(6):
#         rnd = np.random.random(size=(1, 34))
#         if zio == "rec":
#             print("zio")
#             _[i] = rnd
#         else:
#             print('lig')
#             _a[i] = rnd
#
# print(_.shape)
# print(_a.shape)
# print(_)
# #
# # x = 222
# # _ = [2, 22, 222]
# # if x in _:
# #     print(x, _.index(x))
#
# #
# # eleTypes = ['N.3', "N.am", "N.4", 'C.3', 'C.2', 'O.2', 'O.3', 'O.co2']
# # amino_acids = ["ALA", "ARG", "ASN", "ASP", "CYS", "CYX", "GLN", "GLU", "GLY", "HIS", "HIE",
# #                "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
# # accepted_atoms = ['N', 'CA', 'CB', 'C', 'O']
# #
# # print(len(eleTypes), len(amino_acids), len(accepted_atoms))
# #
# # _ = np.array([1, 2, 3])
# # x = 99
# #
# # a = np.hstack([_, x])
# # print(a)
#
# _ = ['E', 'D', 'A  C', 'protein | peptide']
# if any(info == "" or info is None for info in _):
#     print("NO")


import numpy as np

_ = np.array([[1, 2, 3], [4, 5, 6]])
padded = np.vstack((_, np.zeros(shape=(3, 3))))

_1 = np.array([[1, 2, 3], [4, 5, 6]])
padded1 = np.vstack((_, np.zeros(shape=(6, 3))))

print(padded.shape, padded1.shape)

label = np.array(-45, )
