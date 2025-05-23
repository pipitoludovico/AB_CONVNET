from os import path, makedirs
import sys
from subprocess import run
from collections import defaultdict
import gc
import json

import numpy as np
import MDAnalysis as Mda

np.set_printoptions(threshold=sys.maxsize)

matrixData = []

ELE_TYPES = ['N.3', "N.am", "N.4", 'C.3', 'C.2', 'O.2', 'O.3', 'O.co2']
AMINO_ACIDS = ["ALA", "ARG", "ASN", "ASP", "CYS", "CYX", "GLN", "GLU", "GLY",
               "HIS", "HIE", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
               "THR", "TRP", "TYR", "VAL"]
ACCEPTED_ATOMS = ['N', 'CA', 'CB', 'C', 'O']


def Sampler(path_) -> None:
    if path.isdir(path_):
        with open('samples_to_test.json', 'r') as jsonFile:
            data = json.load(jsonFile)
        for entry in data:
            file_PATH = str(path.join(path_, entry["name"] + ".pdb"))
            if path.exists(file_PATH):
                print("\nProcessing", file_PATH)
                try:
                    hChain, lChain, agChain = entry['H_Chain'], entry['L_Chain'], entry['AG_Chain']
                    WriteInterface(file_PATH, hChain, lChain, agChain)
                    BuildMatrix(file_PATH)
                except Exception as e:
                    print(f"{file_PATH} failed. Skipping.\n{e}\nMoving to the next file.")
    else:
        raise FileNotFoundError(
            "Please set the path where the pdb files are contained and make sure that the json is in your cwd.")


def WriteInterface(selected_path, hChain_, lChain_, agChain_) -> None:
    """
    Creates rec.mol2, lig.mol2 and interface.pdb.
    """
    idPath = selected_path.split("/")[-1].replace(".pdb", "")
    makedirs('./predictions', exist_ok=True)
    makedirs(f'./predictions/{idPath}', exist_ok=True)
    u = Mda.Universe(f'{selected_path}')

    try:
        ag_neighbors = u.select_atoms(f"same residue as (protein and around 5 chainID {agChain_})")
        hl_neighbors = u.select_atoms(
            f"same residue as (protein and around 5 (chainID {hChain_} or chainID {lChain_}))")
        ag_neighbors.write(f'./predictions/{idPath}/rec.pdb')
        hl_neighbors.write(f'./predictions/{idPath}/lig.pdb')
        interface = ag_neighbors + hl_neighbors
        interface.write(f'./predictions/{idPath}/interface.pdb')
    finally:
        del u
        gc.collect()  # Force garbage collection to close file handles because MDA keeps the files open -.-

    for structure in ["interface.pdb", 'rec.pdb', 'lig.pdb']:
        print(f"Doing {structure} in {selected_path}")  # ./test/8jx3.pdb
        mol2_filename = structure.split("/")[-1].replace('pdb', 'mol2')
        mol2_path = f"./predictions/{idPath}/{mol2_filename}"
        print(f"MOL2PATH {mol2_path}")
        if not path.exists(mol2_path) or path.getsize(mol2_path) == 0:
            run(f"obabel -i pdb ./predictions/{idPath}/{structure} -o mol2 -O {mol2_path} --partialcharge eem -xs",
                shell=True)


def BuildMatrix(selected_path: str) -> np.array:
    eleTypes = ['N.3', "N.am", "N.4", 'C.3', 'C.2', 'O.2', 'O.3', 'O.co2']
    amino_acids = ["ALA", "ARG", "ASN", "ASP", "CYS", "CYX", "GLN", "GLU", "GLY", "HIS", "HIE",
                   "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
    accepted_atoms = ['N', 'CA', 'CB', 'C', 'O']

    # Load the global feature scaler
    # feature_scaler = joblib.load("feature_scaler.pkl")

    idPath = selected_path.split("/")[-1].replace(".pdb", "")

    def build_residue_map(filename: str) -> dict:
        residue_map = defaultdict(list)
        with open(filename, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) < 9:
                    continue
                atom_name = parts[1]
                res_full = parts[7]
                res_name, res_num = res_full[:3], res_full[3:]
                if atom_name in accepted_atoms:
                    residue_map[(res_name, res_num)].append((atom_name, line))
        return residue_map

    def process_interface(interface_file: str, target_map: dict) -> np.ndarray:
        residue_data = defaultdict(dict)

        with open(interface_file, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) < 9:
                    continue

                atom_name = parts[1]
                res_full = parts[7]
                res_name, res_num = res_full[:3], res_full[3:]
                key = (res_name, res_num)

                if atom_name in accepted_atoms and key in target_map:
                    x, y, z = map(float, parts[2:5])
                    charge = float(parts[8])
                    atom_type = parts[5]

                    # Apply scaling to (x, y, z)
                    # coords_scaled = feature_scaler.transform([[x, y, z]])[0]

                    # One-hot encoding
                    atom_type_onehot = np.zeros(len(eleTypes))
                    if atom_type in eleTypes:
                        atom_type_onehot[eleTypes.index(atom_type)] = 1.0

                    residue_onehot = np.zeros(len(amino_acids))
                    if res_name in amino_acids:
                        residue_onehot[amino_acids.index(res_name)] = 1.0

                    # Feature vector
                    features = np.hstack([x, y, z, charge, atom_type_onehot, residue_onehot])
                    residue_data[key][atom_name] = features

        X = len(residue_data)
        matrix = np.zeros((X, len(accepted_atoms), 34))

        for i, (res_key, atoms) in enumerate(residue_data.items()):
            for j, atom_name in enumerate(accepted_atoms):
                if atom_name in atoms:
                    matrix[i, j] = atoms[atom_name]

        return matrix

    def SaveResults(save_dir: str, abMatrix_: np.ndarray, agMatrix_: np.ndarray, suffix=""):
        print(f"Saving abMatrix {abMatrix_.shape} abMatrix{suffix}")
        np.save(path.join(save_dir, f"abMatrix{suffix}.npy"), abMatrix_)
        print(f"Saving agMatrix {agMatrix_.shape} agMatrix{suffix}")
        np.save(path.join(save_dir, f"agMatrix{suffix}.npy"), agMatrix_)
        print("")

    rec_map = build_residue_map(f'./predictions/{idPath}/rec.mol2')
    lig_map = build_residue_map(f'./predictions/{idPath}/lig.mol2')

    abMatrix = process_interface(f'./predictions/{idPath}/interface.mol2', rec_map)
    agMatrix = process_interface(f'./predictions/{idPath}/interface.mol2', lig_map)
    SaveResults(f"./predictions/{idPath}", abMatrix, agMatrix, idPath)

    print(f"original abMatrix shape: {abMatrix.shape}")
    print(f"original agMatrix shape: {agMatrix.shape}")
