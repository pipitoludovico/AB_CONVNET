import os.path
from os import cpu_count
from multiprocessing import Pool
from subprocess import run
import numpy as np
import MDAnalysis as Mda
import logging
from collections import defaultdict
from scipy.spatial.transform import Rotation
import gc


class FeaturizerClass:
    def __init__(self, dbDict: dict, root: str):
        self.dbDict = dbDict  # dict[pdb] = [row.Hchain, row.Lchain, row.antigen_chain, row.antigen_type]
        self.root = root
        self.GBSA = 0

    def GetGBSA(self, selected_path):
        gbsa = 0
        with open(f'{selected_path}/results_mmgbsa.dat', 'r') as GBSAfile:
            for line in GBSAfile.readlines():
                if "DELTA TOTAL" in line:
                    gbsa = float(line.split()[2])
        self.GBSA = gbsa

    def Featurize(self, pdbID: str, chains_and_type: list) -> None:
        pdbID = pdbID.strip().lower()
        selected_path = os.path.join(self.root, "selected", pdbID)
        if not os.path.exists(selected_path) or any(info == "" or info is None for info in chains_and_type):
            return
        else:
            hChain, lChain, agChain = chains_and_type[0], chains_and_type[1], chains_and_type[2]
            print(f"\nWorking in {selected_path} {hChain} {lChain} {agChain}")
            try:
                print(f"before running write interface for {pdbID}")
                self.WriteInterface(selected_path, hChain, lChain,
                                    agChain)  # scrivo l'interfaccia, e poi anche rec e lig
                print(f"Inferface written in {selected_path}")
                self.GetGBSA(selected_path)  # creo la label...
                print(f"GBSA for {selected_path}: {self.GBSA}")
            except Exception as e:
                print(repr(e))
                gc.collect()
            if self.GBSA < 0:
                self.BuildMatrix(pdbID, selected_path)  # , dec_res)

    @staticmethod
    def WriteInterface(selected_path, hChain_, lChain_, agChain_) -> None:
        """
        Creates rec.mol2, lig.mol2 and interface.pdb.
        """
        u = Mda.Universe(f'{selected_path}/complex_minimized_chains.pdb')

        try:
            ag_neighbors = u.select_atoms(f"same residue as (protein and around 5 chainID {agChain_})")
            hl_neighbors = u.select_atoms(f"same residue as (protein and around 5 (chainID {hChain_} or chainID {lChain_}))")
            ag_neighbors.write(f'{selected_path}/rec.pdb')
            hl_neighbors.write(f'{selected_path}/lig.pdb')
            interface = ag_neighbors + hl_neighbors
            interface.write(f'{selected_path}/interface.pdb')

        finally:
            del u
            gc.collect()  # Force garbage collection to close file handles

        for structure in ["rec.pdb", "lig.pdb", "interface.pdb"]:
            print(f"Doing {structure} in {selected_path}")
            mol2_filename = structure.replace('pdb', 'mol2')
            mol2_path = f"{selected_path}/{mol2_filename}"
            if not os.path.exists(mol2_path):
                run(f"obabel -i pdb {selected_path}/{structure} -o mol2 -O {mol2_path} --partialcharge eem -xs", shell=True)

    def BuildMatrix(self, pdbID: str, selected_path: str) -> np.array:
        eleTypes = ['N.3', "N.am", "N.4", 'C.3', 'C.2', 'O.2', 'O.3', 'O.co2']
        amino_acids = ["ALA", "ARG", "ASN", "ASP", "CYS", "CYX", "GLN", "GLU", "GLY", "HIS", "HIE",
                       "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
        accepted_atoms = ['N', 'CA', 'CB', 'C', 'O']

        def build_residue_map(filename: str) -> dict:
            """Create a residue map for a given file"""
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

        def GenerateAugmentedMatrices(selected_path_: str,
                                      abMatrix_: np.ndarray,
                                      agMatrix_: np.ndarray,
                                      gbsa: float,
                                      n_augmentations: int = 5) -> None:
            """Generate augmented versions of the matrices with random rotations"""
            for i in range(n_augmentations):
                # Generate random rotation matrix (same for both matrices)
                rotation = Rotation.random().as_matrix()
                # Apply rotation to coordinates of both matrices
                aug_ab = apply_rotation(abMatrix_.copy(), rotation)
                aug_ag = apply_rotation(agMatrix_.copy(), rotation)
                # Save augmented versions
                SaveResults(selected_path_, aug_ab, aug_ag, gbsa, f"_aug_{i}")
                del aug_ag, aug_ab

        def apply_rotation(matrix: np.ndarray, rotation: np.ndarray) -> np.ndarray:
            """Apply rotation to the x,y,z coordinates in the matrix"""
            # Coordinates are assumed to be the first 3 features
            coords = matrix[..., :3]  # Shape: (X, 5, 3)
            rotated_coords = np.dot(coords, rotation.T)  # Apply rotation to all coordinates

            # Replace original coordinates with rotated ones
            rotated_matrix = matrix.copy()
            rotated_matrix[..., :3] = rotated_coords
            del matrix
            return rotated_matrix

        def process_interface(interface_file: str, target_map: dict) -> np.ndarray:
            """Process interface file against target map and return matrix"""
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
                        atom_type = parts[5]
                        # charge = float(parts[8])

                        atom_type_onehot = np.zeros(len(eleTypes))
                        if atom_type in eleTypes:
                            atom_type_onehot[eleTypes.index(atom_type)] = 1.0

                        residue_onehot = np.zeros(len(amino_acids))
                        if res_name in amino_acids:
                            residue_onehot[amino_acids.index(res_name)] = 1.0

                        atom_name_onehot = np.zeros(len(accepted_atoms))
                        atom_name_onehot[accepted_atoms.index(atom_name)] = 1.0

                        # features = np.hstack([x, y, z, charge, atom_type_onehot, residue_onehot])
                        features = np.hstack([x, y, z, atom_name_onehot, residue_onehot])
                        residue_data[key][atom_name] = features

            # Converto ad array su quanti residui ci sono
            X = len(residue_data)
            matrix = np.zeros((X, len(accepted_atoms), (3 + len(atom_name_onehot) + len(residue_onehot))))

            for i, (res_key, atoms) in enumerate(residue_data.items()):
                for j, atom_name in enumerate(accepted_atoms):
                    if atom_name in atoms:
                        matrix[i, j] = atoms[atom_name]

            return matrix

        def SaveResults(selected_path_: str, abMatrix_: np.ndarray, agMatrix_: np.ndarray, gbsa: float, suffix=""):
            """Save matrices and GBSA value to disk"""
            # Create saved_results directory if it doesn't exist
            save_dir = os.path.join(selected_path_, "saved_results")
            os.makedirs(save_dir, exist_ok=True)

            # Save matrices as numpy files
            np.save(os.path.join(save_dir, f"abMatrix{suffix}.npy"), abMatrix_)
            np.save(os.path.join(save_dir, f"agMatrix{suffix}.npy"), agMatrix_)
            np.save(os.path.join(save_dir, "label.npy"), gbsa)
            print(f"Saving abMatrix {abMatrix_.shape} abMatrix{suffix}")
            print(f"Saving agMatrix {agMatrix_.shape} agMatrix{suffix}")
            print("")

        # Build matrices
        rec_map = build_residue_map(f'{selected_path}/rec.mol2')
        lig_map = build_residue_map(f'{selected_path}/lig.mol2')

        abMatrix = process_interface(f'{selected_path}/interface.mol2', rec_map)
        agMatrix = process_interface(f'{selected_path}/interface.mol2', lig_map)

        print(f"Generated matrices for {pdbID}:")
        print(f"original abMatrix shape: {abMatrix.shape}")
        print(f"original agMatrix shape: {agMatrix.shape}")
        print(f"GBSA: {self.GBSA}\n")
        SaveResults(selected_path, abMatrix, agMatrix, self.GBSA)
        GenerateAugmentedMatrices(selected_path, abMatrix, agMatrix, self.GBSA)
        del abMatrix, agMatrix


def ParallelFeaturize(dbDict, root) -> None:
    featurizer = FeaturizerClass(dbDict, root)
    cpuUnits = int(cpu_count() // 4)
    with Pool(processes=cpuUnits) as p:
        for pdbID, chains_and_type in dbDict.items():
            p.apply_async(featurizer.Featurize, args=(pdbID, chains_and_type,))
        p.close()
        p.join()
    logging.debug("All processes have completed.")
