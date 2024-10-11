from os import makedirs, listdir
from os.path import isdir, isfile, exists
from subprocess import run
import numpy as np

from include.Interfacer.VMD import GetVDWcontacts

matrixData = []


def GetFeatures(path_) -> list:
    if isdir(path_):
        for idx, file in enumerate(listdir(path_)):
            if file.endswith(".pdb"):
                matrixData.append(BuildMatrix(idx, path_))
    if isfile(path_):
        matrixData.append(BuildMatrix(0, path_))
    return matrixData


def BuildMatrix(idx: int, path_: str) -> np.array:
    elements = {'H': 1.008, 'He': 4.002, 'Li': 6.941, 'Be': 9.012, 'B': 10.811, 'C': 12.011, 'N': 14.007,
                'O': 15.999, 'F': 18.998, 'Ne': 20.180, 'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.085,
                'P': 30.974, 'S': 32.066, 'Cl': 35.453, 'Ar': 39.948, 'K': 39.098, 'Ca': 40.078, 'Sc': 44.956,
                'Ti': 47.867, 'V': 50.942, 'Cr': 51.996, 'Mn': 54.938, 'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693,
                'Cu': 63.546, 'Zn': 65.38, 'Ga': 69.723, 'Ge': 72.64, 'As': 74.922, 'Se': 78.96, 'Br': 79.904,
                'Kr': 83.798, 'Rb': 85.468, 'Sr': 87.62, 'Y': 88.906, 'Zr': 91.224, 'Nb': 92.906, 'Mo': 95.96,
                'Tc': 98.0, 'Ru': 101.07, 'Rh': 102.906, 'Pd': 106.42, 'Ag': 107.868, 'Cd': 112.411, 'In': 114.818,
                'Sn': 118.71, 'Sb': 121.76, 'Te': 127.6, 'I': 126.904, 'Xe': 131.293, 'Cs': 132.905, 'Ba': 137.327,
                'La': 138.905, 'Ce': 140.116, 'Pr': 140.908, 'Nd': 144.242, 'Pm': 145.0, 'Sm': 150.36,
                'Eu': 151.964, 'Gd': 157.25, 'Tb': 158.925, 'Dy': 162.5, 'Ho': 164.930, 'Er': 167.259, 'Tm': 168.934,
                'Yb': 173.04,
                'Lu': 174.967, 'Hf': 178.49, 'Ta': 180.948, 'W': 183.84, 'Re': 186.207, 'Os': 190.23, 'Ir': 192.217,
                'Pt': 195.084, 'Au': 196.967, 'Hg': 200.59, 'Tl': 204.383, 'Pb': 207.2, 'Bi': 208.980,
                'Th': 232.038, 'Pa': 231.036, 'U': 238.029}
    finalCoordinates = {}
    couples = []
    resultFilePath: str = f"predictions/{path_.replace('.pdb', '')}"

    makedirs("predictions", exist_ok=True)
    makedirs(f"{resultFilePath}", exist_ok=True)
    # we get the contact pairs and create .int file
    intPath: str = f"{resultFilePath}/{path_.replace('.pdb', '.int')}"
    mol2Path: str = f"{resultFilePath}/{path_.replace('.pdb', '.mol2')}"
    GetVDWcontacts(filePath=path_, abChains="H L", agChains="not H L", outputPath=intPath)
    # then we make the mol2 with coordinates, charges and type
    if not exists(f"{mol2Path}"):
        run(f"obabel -i pdb {path_} -o mol2 -O {mol2Path} > logs/obabelLog.log", shell=True)
    # we check contact points between Ab and Ag and make the pairs
    with open(intPath, 'r') as contacts:
        contact_content = contacts.read()
        results = contact_content.replace("\n", '').split(',')[1:]
        for couple in results:
            couples.append(couple)
    # and we add, if any, the % of molecular surface patch % to the score. This is where we build the final matrix.
    with open(mol2Path, 'r') as complexMOL2:
        finalCoordinates[idx] = {}
        for line in complexMOL2.readlines():
            if len(line.split()) > 5:
                try:
                    x, y, z, atomType, partialCharge, ResidAndResnum = line.split()[2], line.split()[3], line.split()[
                        4], line.split()[5], line.split()[8], line.split()[7]
                    x, y, z, partialCharge = map(float, [x, y, z, partialCharge])
                    if ResidAndResnum not in finalCoordinates[idx]:
                        finalCoordinates[idx][ResidAndResnum] = []
                        finalCoordinates[idx][ResidAndResnum].append(
                            np.array([elements[atomType.split(".")[0]], partialCharge, x, y, z]))
                    else:
                        finalCoordinates[idx][ResidAndResnum].append(
                            np.array([elements[atomType.split(".")[0]], partialCharge, x, y, z]))
                except IndexError:
                    print(f"MOLECULE {idx} HAD AN INDEX OUT OF RANGE IN THE MOL2 FILE.")

    dataList = []
    for pair in couples:
        res1, res2, = pair.split("-")[0], pair.split("-")[1]
        arr1, arr2 = np.array(finalCoordinates[idx][res1]), np.array(finalCoordinates[idx][res2])
        arrayList = [arr1, arr2]
        max_len = max([arr.shape[0] for arr in arrayList])
        # we pad before stacking
        if arr1.shape[0] < arr2.shape[0]:
            paddedarr1 = np.pad(arr1, ((0, max_len - arr1.shape[0]), (0, 0)), 'constant', constant_values=0)
            paddedarr2 = arr2
        else:
            paddedarr1 = arr1
            paddedarr2 = np.pad(arr2, ((0, max_len - arr2.shape[0]), (0, 0)), 'constant', constant_values=0)
        # we stack a squared matrix that contains padded sequences
        a3 = np.hstack((paddedarr1, paddedarr2))
        zero_count = np.sum(a3 == 0, axis=1)
        # then we filter those rows with too many zeroes to trim the matrix => empty rows just to make it square
        filtered_arr = a3[zero_count < 6]
        dataList.append(filtered_arr)
    return np.vstack(dataList)
