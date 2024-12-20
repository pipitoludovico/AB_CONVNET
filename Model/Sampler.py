import sys
import os
from os import makedirs, listdir
from os.path import isdir, isfile, exists
from subprocess import run
import numpy as np
from sys import maxsize
from include.Interfacer.VMD import GetVDWcontacts

np.set_printoptions(threshold=maxsize)

matrixData = []


def GetFeatures(path_) -> list:
    if isdir(path_):
        for idx, file in enumerate(listdir(path_)):
            if file.endswith(".pdb"):
                print("\nDoing ", file)
                matrixData.append((BuildMatrix(idx, path_ + file), file))
    if isfile(path_):
        print("\nDoing ", path_)
        matrixData.append((BuildMatrix(0, path_), path_))
    return matrixData


def GetChains(path_):
    chains = []
    remarks = False
    _HCHAIN, _LCHAIN, _AGCHAIN = "", "", ""
    with open(path_, 'r') as pdb:
        for line in pdb.readlines():
            if 'PAIRED_HL' in line:
                remarks = True
                _HCHAIN = line.split()[3].split("=")[1]
                _LCHAIN = line.split()[4].split("=")[1]
                _AGCHAIN = line.split()[5].split("=")[1]
            else:
                if 'ATOM' in line:
                    if line.split()[4].strip() not in chains:
                        chains.append(line.split()[4].strip())
    if remarks:
        return _HCHAIN, _LCHAIN, _AGCHAIN
    elif len(chains) > 2:
        print("getting chains from the pdb.")
        return chains[0], chains[1], chains[2]
    else:
        print("No chain found in the REMARK. Will try using H, L, A as default.")
        return "H", "L", "A"


def BuildMatrix(idx: int, path_: str) -> np.array:
    _HCHAIN, _LCHAIN, _AGCHAIN = "H", "L", "A"
    try:
        _HCHAIN, _LCHAIN, _AGCHAIN = GetChains(path_)
        print("Spotted chains for ", path_, _HCHAIN, _LCHAIN, _AGCHAIN)
    except Warning:
        print("Could not find the chains. Trying with the default H L A.")
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
    amino_acids = ["DAMP", "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
                   "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", 'CYX', "HIE"]
    finalCoordinates = {}
    couples = []
    fileName = path_.split("/")[-1]
    resultFilePath: str = f"predictions/{path_.replace('.pdb', '')}"

    makedirs("predictions", exist_ok=True)
    makedirs(f"{resultFilePath}", exist_ok=True)
    # we get the contact pairs and create .int file
    intPath: str = f"{resultFilePath}/{fileName.replace('.pdb', '.int')}"
    mol2Path: str = f"{resultFilePath}/{fileName.replace('.pdb', '.mol2')}"
    makedirs('logs', exist_ok=True)
    try:
        GetVDWcontacts(filePath=path_, abChains=" ".join([_HCHAIN, _LCHAIN]), agChains=" ".join(_AGCHAIN),
                       outputPath=intPath)
        # then we make the mol2 with coordinates, charges and type
        if not exists(f"{mol2Path}"):
            run(f"obabel -i pdb {path_} -o mol2 -O {mol2Path} > {resultFilePath}/obabelLog.log", shell=True)
        # we check contact points between Ab and Ag and make the pairs
        with open(intPath, 'r') as contacts:
            contact_content = contacts.read()
            results = contact_content.replace("\n", '').split(',')[1:]
            # numContacts = int(contact_content.split(",")[0])
            for couple in results:
                couples.append(couple)
        # and we add, if any, the % of molecular surface patch % to the score. This is where we build the final matrix.
        with open(mol2Path, 'r') as complexMOL2:
            finalCoordinates[idx] = {}
            for line in complexMOL2.readlines():
                if len(line.split()) > 5:
                    if line.split()[1] in ['N', 'CA', 'CB', 'C', 'O']:
                        try:
                            x, y, z, atomType, partialCharge, Resid, Resnum = line.split()[2], line.split()[3], \
                                line.split()[4], line.split()[5], line.split()[8], line.split()[7][0:3], \
                                line.split()[7][3:]
                            x, y, z, partialCharge = map(float, [x, y, z, partialCharge])
                            # prob = float(tempPesto[pdbID].get(ResidAndResnum, 0))
                            ResidAndResnum = f"{Resid}{Resnum}"
                            atomMass = elements[atomType.split(".")[0]]
                            if ResidAndResnum not in finalCoordinates[idx]:
                                finalCoordinates[idx][ResidAndResnum] = []
                                finalCoordinates[idx][ResidAndResnum].append(
                                    np.array([atomMass, partialCharge, x, y, z]))
                            else:
                                finalCoordinates[idx][ResidAndResnum].append(
                                    np.array([atomMass, partialCharge, x, y, z]))
                                # np.array([elements[atomType.split(".")[0]], partialCharge, x, y, z, prob]))
                        except IndexError:
                            print(idx, "HAD AN INDEX OUT OF RANGE IN THE MOL2 FILE.")

        dataList = []
        for pair in couples[:100]:
            res1, res2, = pair.split("-")[0], pair.split("-")[1]
            # dec1, dec2 = dec_res_.get(res1, 0), dec_res_.get(res2, 0)
            arr1 = np.hstack(finalCoordinates[idx][res1] + [amino_acids.index(res1[0:3])])
            arr2 = np.hstack(finalCoordinates[idx][res2] + [amino_acids.index(res2[0:3])])
            # we pad before stacking
            target_length = 26
            padded_arr1 = np.pad(arr1, (0, target_length - arr1.size), 'constant')
            padded_arr2 = np.pad(arr2, (0, target_length - arr2.size), 'constant')
            a3 = np.hstack((padded_arr1, padded_arr2))
            dataList.append(a3)
        dataMatrix = np.hstack(dataList)
        finalCoordinates.clear()
        # stacked = np.vstack(dataList)
        np.save(f"{resultFilePath}/{fileName}.npy", dataMatrix, allow_pickle=True)
        return dataMatrix
    except Exception as e:
        print(path_, " had incorrect H L chains or contained other errors.")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
