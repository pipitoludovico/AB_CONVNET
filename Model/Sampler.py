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

    # Define atom types and amino acids as in the featurizer
    eleTypes = ['N.3', "N.am", "N.4", 'C.3', 'C.2', 'O.2', 'O.3', 'O.co2']
    amino_acids = ["ALA", "ARG", "ASN", "ASP", "CYS", "CYX", "GLN", "GLU", "GLY", "HIS", "HIE",
                   "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]

    try:
        _HCHAIN, _LCHAIN, _AGCHAIN = GetChains(path_)
        print("Spotted chains for ", path_, _HCHAIN, _LCHAIN, _AGCHAIN)
    except Warning:
        print("Could not find the chains. Trying with the default H L A.")

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
    if any(len(chain) > 1 for chain in [_HCHAIN, _LCHAIN, _AGCHAIN]):
        print(f"Wrong chainIDs for pdb {path_}")
        return None
    else:
        try:
            GetVDWcontacts(filePath=path_, abChains=" ".join([_HCHAIN, _LCHAIN]),
                           agChains=" ".join(_AGCHAIN), outputPath=intPath)

            # then we make the mol2 with coordinates, charges and type
            if not exists(f"{mol2Path}"):
                run(f"obabel -i pdb {path_} -o mol2 -O {mol2Path} > {resultFilePath}/obabelLog.log", shell=True)

            # we check contact points between Ab and Ag and make the pairs
            with open(intPath, 'r') as contacts:
                contact_content = contacts.read()
                results = contact_content.replace("\n", '').split(',')[1:]
                numContacts = int(contact_content.split(",")[0])
                for couple in results:
                    couples.append(couple)

            # Build the feature matrix as in the featurizer
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
                                ResidAndResnum = f"{Resid}{Resnum}"

                                # Create one-hot encodings as in featurizer
                                atom_type_onehot = np.zeros(len(eleTypes))
                                residue_onehot = np.zeros(len(amino_acids))
                                if atomType in eleTypes:
                                    atom_type_onehot[eleTypes.index(atomType)] = 1.0
                                if Resid in amino_acids:
                                    residue_onehot[amino_acids.index(Resid)] = 1.0

                                # Combine features as in featurizer
                                full_feature = np.concatenate(
                                    [[x, y, z, partialCharge], atom_type_onehot, residue_onehot])

                                if ResidAndResnum not in finalCoordinates[idx]:
                                    finalCoordinates[idx][ResidAndResnum] = []
                                finalCoordinates[idx][ResidAndResnum].append(full_feature)

                            except IndexError:
                                print(idx, "HAD AN INDEX OUT OF RANGE IN THE MOL2 FILE.")

            residuePairs = []
            numberOfPairs: int = 5  # Changed from 50 to match the featurizer
            for pair in couples[:numberOfPairs]:
                res1, res2 = pair.split("-")[0], pair.split("-")[1]

                # Get the feature arrays for each residue
                res1array = np.array(finalCoordinates[idx][res1])
                res2array = np.array(finalCoordinates[idx][res2])

                # Pad to ensure 5 atoms per residue (as in featurizer)
                res1Pad = np.pad(res1array, pad_width=((0, 5 - res1array.shape[0]), (0, 0)),
                                 mode='constant', constant_values=0)
                res2Pad = np.pad(res2array, pad_width=((0, 5 - res2array.shape[0]), (0, 0)),
                                 mode='constant', constant_values=0)

                # Stack the pairs horizontally
                stackedCouple = np.hstack([res1Pad, res2Pad])
                reshaped = stackedCouple.reshape(-1)  # Flatten to 1D array
                residuePairs.append(reshaped)

            # Combine all pairs into final matrix
            if residuePairs:
                dataMatrix = np.array(residuePairs).reshape(-1)
                print(f"SHAPE OF {dataMatrix.shape} for {path_}")

                # Save the matrix without label (since we're predicting)
                np.save(f"{resultFilePath}/{fileName}.npy", dataMatrix, allow_pickle=True)
                return dataMatrix
            else:
                print(f"No valid residue pairs found for {path_}")
                return np.array([])

        except Exception as e:
            print(path_, " had incorrect H L chains or contained other errors.")
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            return np.array([])
