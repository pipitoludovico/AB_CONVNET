import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from os import path, listdir, makedirs
from subprocess import run
import MDAnalysis as Mda
import gc
from collections import defaultdict
import pickle


class AntibodyMutator:
    """
    A class that reads antibody-antigen complexes from PDB files,
    feeds them to a cGAN, and returns mutated antibodies.
    """

    hChain, lChain, agChain = "H", "L", "A"
    amino_acids = ["ALA", "ARG", "ASN", "ASP", "CYS", "CYX", "GLN", "GLU", "GLY", "HIS", "HIE",
                   "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
    accepted_atoms = ['N', 'CA', 'CB', 'C', 'O']

    def __init__(self, model_path: str, complex_folder: str, scaler_path: str = None,
                 ab_max_length: int = 92, ag_max_length: int = 97):
        """
        Initialize the AntibodyMutator.

        Args:
            model_path: Path to the trained cGAN generator model
            complex_folder: Folder containing PDB files with Ab-Ag complexes
            scaler_path: Path to saved feature scaler (optional)
            ab_max_length: Maximum length for antibody padding
            ag_max_length: Maximum length for antigen padding
        """
        self.model_path = model_path
        self.complex_folder = complex_folder
        self.scaler_path = scaler_path
        self.ab_max_length = ab_max_length
        self.ag_max_length = ag_max_length

        self.model = None
        self.feature_scaler = None

        self._check_complex_folder()
        self._load_model()
        self._load_scaler()

    def _check_complex_folder(self):
        """Check if complex folder exists and contains files"""
        if path.exists(self.complex_folder):
            if len(listdir(self.complex_folder)) == 0:
                print(
                    f"Please put an Ab-Ag complex inside your {self.complex_folder} folder and make sure that the Ag chain is named A and that Ab Chains are named H and L.")
                exit()
        else:
            print(f"Complex folder {self.complex_folder} does not exist.")
            exit()

    def _load_model(self):
        """Load the trained cGAN generator model"""
        if self.model_path is not None:
            if path.exists(self.model_path):
                try:
                    self.model = load_model(self.model_path, compile=False, safe_mode=False)
                    print(f"Model {self.model_path} successfully loaded.")
                except Exception as e:
                    print(f"Model load failed:\n{e}")
                    exit()
            else:
                print(f"File not found: {self.model_path}")
                exit()
        else:
            print("Please provide the model path you want to use.")
            exit()

    def _load_scaler(self):
        """Load the feature scaler if path is provided"""
        if self.scaler_path and path.exists(self.scaler_path):
            try:
                with open(self.scaler_path, 'rb') as f:
                    self.feature_scaler = pickle.load(f)
                print(f"Feature scaler loaded from {self.scaler_path}")
            except Exception as e:
                print(f"Failed to load scaler: {e}")
                self.feature_scaler = None
        else:
            print("No scaler provided or scaler file not found. Will skip feature scaling.")
            self.feature_scaler = None

    def _write_interface(self, selected_path: str, hChain_: str, lChain_: str, agChain_: str) -> tuple:
        """
        Creates rec.mol2, lig.mol2 and interface.pdb.
        Returns (success, folderID)
        """
        success = True
        idPath = selected_path.split("/")[-1].replace(".pdb", "")
        makedirs('./predictions', exist_ok=True)
        makedirs(f'./predictions/{idPath}', exist_ok=True)
        u = Mda.Universe(f'{selected_path}')

        try:
            ag_neighbors = u.select_atoms(f"same residue as (protein and around 5 chainID {agChain_})")
            hl_neighbors = u.select_atoms(
                f"same residue as (protein and around 5 (chainID {hChain_} or chainID {lChain_}))")
            if len(ag_neighbors) == 0:
                print(f"No antigen atoms found for chain {agChain_}")
                success = False
            if len(hl_neighbors) == 0:
                print(f"No HL atoms found for chains {hChain_} {lChain_}")
                success = False
            ag_neighbors.write(f'./predictions/{idPath}/rec.pdb')
            hl_neighbors.write(f'./predictions/{idPath}/lig.pdb')
            interface = ag_neighbors + hl_neighbors
            interface.write(f'./predictions/{idPath}/interface.pdb')
        finally:
            del u
            gc.collect()

        for structure in ["rec.pdb", "lig.pdb", "interface.pdb"]:
            print(f"Processing {structure} in {selected_path}")
            input_pdb_path = f"predictions/{idPath}/{structure}"
            output_mol2_path = f"predictions/{idPath}/{structure.replace('pdb', 'mol2')}"
            if not path.exists(output_mol2_path):
                print(f"Building the mol2 for {output_mol2_path}")
                run(f"obabel -i pdb {input_pdb_path} -o mol2 -O {output_mol2_path} --partialcharge eem -xs", shell=True)

        return success, idPath

    def _build_residue_map(self, filename: str) -> dict:
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
                if atom_name in self.accepted_atoms:
                    residue_map[(res_name, res_num)].append((atom_name, line))
        return residue_map

    def _process_interface(self, interface_file: str, target_map: dict) -> np.ndarray:
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

                if atom_name in self.accepted_atoms and key in target_map:
                    x, y, z = map(float, parts[2:5])

                    residue_onehot = np.zeros(len(self.amino_acids))
                    if res_name in self.amino_acids:
                        residue_onehot[self.amino_acids.index(res_name)] = 1.0

                    atom_name_onehot = np.zeros(len(self.accepted_atoms))
                    atom_name_onehot[self.accepted_atoms.index(atom_name)] = 1.0

                    features = np.hstack([x, y, z, atom_name_onehot, residue_onehot])
                    residue_data[key][atom_name] = features

        # Convert to matrix
        X = len(residue_data)
        matrix = np.zeros((X, len(self.accepted_atoms), (3 + len(self.accepted_atoms) + len(self.amino_acids))))

        for i, (res_key, atoms) in enumerate(residue_data.items()):
            for j, atom_name in enumerate(self.accepted_atoms):
                if atom_name in atoms:
                    matrix[i, j] = atoms[atom_name]

        return matrix

    def _pad_matrix(self, matrix: np.ndarray, max_length: int) -> np.ndarray:
        """Pad matrix to max_length with zeros"""
        if matrix.shape[0] > max_length:
            print(f"Warning: Matrix length {matrix.shape[0]} exceeds max_length {max_length}. Truncating.")
            return matrix[:max_length]
        elif matrix.shape[0] < max_length:
            padding = np.zeros((max_length - matrix.shape[0], matrix.shape[1], matrix.shape[2]))
            return np.vstack([matrix, padding])
        return matrix

    def _apply_scaling(self, ab_matrix: np.ndarray, ag_matrix: np.ndarray) -> tuple:
        """Apply feature scaling to the matrices"""
        if self.feature_scaler is None:
            return ab_matrix, ag_matrix

        # Apply scaling only to x, y, z coordinates (first 3 features)
        continuous_idx = slice(0, 3)

        # Create masks to identify non-padded entries
        ab_mask = np.any(ab_matrix != 0, axis=-1)
        ag_mask = np.any(ag_matrix != 0, axis=-1)

        # Extract continuous features
        ab_cont = ab_matrix[..., continuous_idx]
        ag_cont = ag_matrix[..., continuous_idx]

        # Apply scaling only to non-padded entries
        ab_cont_scaled = ab_cont.copy()
        if np.any(ab_mask):
            ab_cont_scaled[ab_mask] = self.feature_scaler.transform(ab_cont[ab_mask])

        ag_cont_scaled = ag_cont.copy()
        if np.any(ag_mask):
            ag_cont_scaled[ag_mask] = self.feature_scaler.transform(ag_cont[ag_mask])

        # Update original matrices
        ab_matrix_scaled = ab_matrix.copy()
        ag_matrix_scaled = ag_matrix.copy()
        ab_matrix_scaled[..., continuous_idx] = ab_cont_scaled
        ag_matrix_scaled[..., continuous_idx] = ag_cont_scaled

        return ab_matrix_scaled, ag_matrix_scaled

    def encode_complex(self, pdb_path: str) -> tuple:
        """
        Encode a PDB complex into antibody and antigen matrices.

        Args:
            pdb_path: Path to the PDB file

        Returns:
            Tuple of (ab_matrix, ag_matrix, pdb_id)
        """
        success, pdb_id = self._write_interface(pdb_path, self.hChain, self.lChain, self.agChain)

        if not success:
            return None, None, pdb_id

        # Build matrices
        rec_map = self._build_residue_map(f'predictions/{pdb_id}/rec.mol2')
        lig_map = self._build_residue_map(f'predictions/{pdb_id}/lig.mol2')

        ab_matrix = self._process_interface(f'predictions/{pdb_id}/interface.mol2', lig_map)
        ag_matrix = self._process_interface(f'predictions/{pdb_id}/interface.mol2', rec_map)

        print(f"Generated matrices for {pdb_id}:")
        print(f"Original ab_matrix shape: {ab_matrix.shape}")
        print(f"Original ag_matrix shape: {ag_matrix.shape}")

        # Pad matrices to expected lengths
        ab_matrix = self._pad_matrix(ab_matrix, self.ab_max_length)
        ag_matrix = self._pad_matrix(ag_matrix, self.ag_max_length)

        print(f"Padded ab_matrix shape: {ab_matrix.shape}")
        print(f"Padded ag_matrix shape: {ag_matrix.shape}")

        # Apply feature scaling if available
        ab_matrix, ag_matrix = self._apply_scaling(ab_matrix, ag_matrix)

        print(f"CHECK MATRIX AB per {pdb_id}\n {ab_matrix[0][0]}")
        print(f"CHECK MATRIX AG per {pdb_id}\n {ag_matrix[0][0]}")

        return ab_matrix, ag_matrix, pdb_id

    def generate_mutated_antibody(self, ab_matrix: np.ndarray, ag_matrix: np.ndarray) -> np.ndarray:
        """
        Generate a mutated antibody using the cGAN.

        Args:
            ab_matrix: Antibody matrix (shape: ab_max_length, 5, features)
            ag_matrix: Antigen matrix (shape: ag_max_length, 5, features)

        Returns:
            Mutated antibody matrix with same shape as input
        """
        # Add batch dimension
        ab_batch = np.expand_dims(ab_matrix, axis=0)
        ag_batch = np.expand_dims(ag_matrix, axis=0)

        # Generate mutated antibody
        batch_size = tf.shape(ab_batch)[0]
        z = tf.random.normal(shape=(batch_size, 128), dtype=ab_batch.dtype)

        mutated_ab = self.model.predict([ab_batch, ag_batch, z], verbose=1)

        # Remove batch dimension
        return mutated_ab[0]

    def _decode_matrix_to_pdb(self, matrix: np.ndarray, chain_id: str, output_path: str,
                              original_pdb_path: str = None):
        """
        Decode a matrix back to PDB format.

        Args:
            matrix: Matrix to decode (shape: max_length, 5, features)
            chain_id: Chain identifier for the PDB
            output_path: Path to save the PDB file
            original_pdb_path: Optional path to original PDB for template info
        """
        pdb_lines = []
        atom_counter = 1
        residue_counter = 1

        # Process each residue in the matrix
        for res_idx in range(matrix.shape[0]):
            residue_atoms = matrix[res_idx]  # Shape: (5, features)

            # Check if this residue has any non-zero data (not padded)
            if np.all(residue_atoms == 0):
                continue

            # Extract residue type from one-hot encoding
            # Features: [x, y, z, atom_name_onehot(5), residue_onehot(22)]
            residue_onehot = residue_atoms[0, 8:]  # Skip x,y,z and atom_name_onehot
            residue_type_idx = np.argmax(residue_onehot)
            residue_type = self.amino_acids[residue_type_idx] if residue_type_idx < len(self.amino_acids) else "UNK"

            # Process each atom in the residue
            for atom_idx in range(len(self.accepted_atoms)):
                atom_data = residue_atoms[atom_idx]

                # Check if atom exists (has non-zero coordinates)
                if np.all(atom_data[:3] == 0):
                    continue

                # Extract coordinates (reverse scaling if needed)
                x, y, z = atom_data[:3]

                # If we have a scaler, inverse transform the coordinates
                if self.feature_scaler is not None:
                    coords = np.array([[x, y, z]])
                    coords_unscaled = self.feature_scaler.inverse_transform(coords)[0]
                    x, y, z = coords_unscaled

                # Get atom name
                atom_name = self.accepted_atoms[atom_idx]

                # Create PDB line
                pdb_line = f"ATOM  {atom_counter:5d}  {atom_name:<3s} {residue_type:>3s} {chain_id}{residue_counter:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           {atom_name[0]:>1s}\n"
                pdb_lines.append(pdb_line)
                atom_counter += 1

            residue_counter += 1

        # Write PDB file
        with open(output_path, 'w') as f:
            f.write("HEADER    GENERATED BY ANTIBODY MUTATOR\n")
            f.writelines(pdb_lines)
            f.write("END\n")

        print(f"Decoded matrix to PDB: {output_path}")

    def _create_complex_pdb(self, ab_matrix: np.ndarray, ag_matrix: np.ndarray,
                            output_path: str, original_pdb_path: str = None):
        """
        Create a complete antibody-antigen complex PDB file.

        Args:
            ab_matrix: Antibody matrix
            ag_matrix: Antigen matrix
            output_path: Path to save the complex PDB
            original_pdb_path: Optional path to original PDB for reference
        """
        pdb_lines = []
        atom_counter = 1

        # Helper function to process a matrix for a specific chain
        def process_matrix_for_chain(matrix, chain_id):
            nonlocal atom_counter
            chain_lines = []
            residue_counter = 1

            for res_idx in range(matrix.shape[0]):
                residue_atoms = matrix[res_idx]

                # Check if this residue has any non-zero data
                if np.all(residue_atoms == 0):
                    continue

                # Extract residue type from one-hot encoding
                residue_onehot = residue_atoms[0, 8:]
                residue_type_idx = np.argmax(residue_onehot)
                residue_type = self.amino_acids[residue_type_idx] if residue_type_idx < len(self.amino_acids) else "UNK"
                # Process each atom in the residue
                for atom_idx in range(len(self.accepted_atoms)):
                    atom_data = residue_atoms[atom_idx]

                    # Check if atom exists
                    if np.all(atom_data[:3] == 0):
                        continue

                    # Extract coordinates
                    x, y, z = atom_data[:3]

                    # Inverse transform if scaler is available
                    if self.feature_scaler is not None:
                        coords = np.array([[x, y, z]])
                        coords_unscaled = self.feature_scaler.inverse_transform(coords)[0]
                        x, y, z = coords_unscaled

                    # Get atom name
                    atom_name = self.accepted_atoms[atom_idx]

                    # Create PDB line
                    pdb_line = f"ATOM  {atom_counter:5d}  {atom_name:<3s} {residue_type:>3s} {chain_id}{residue_counter:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           {atom_name[0]:>1s}\n"
                    chain_lines.append(pdb_line)
                    atom_counter += 1

                residue_counter += 1

            return chain_lines

        # Process antibody (assuming it contains both H and L chains combined)
        # For simplicity, we'll assign all antibody residues to chain H
        # You might want to modify this based on your specific needs
        pdb_lines.append("HEADER    MUTATED ANTIBODY-ANTIGEN COMPLEX\n")

        # Process antibody matrix
        ab_lines = process_matrix_for_chain(ab_matrix, "H")
        pdb_lines.extend(ab_lines)

        # Add chain terminator
        pdb_lines.append("TER\n")

        # Process antigen matrix
        ag_lines = process_matrix_for_chain(ag_matrix, "A")
        pdb_lines.extend(ag_lines)

        # Add final terminator
        pdb_lines.append("TER\n")
        pdb_lines.append("END\n")

        # Write complex PDB file
        with open(output_path, 'w') as f:
            f.writelines(pdb_lines)

        print(f"Created complex PDB: {output_path}")

    def mutate_complex(self, pdb_path: str, output_dir: str = "./mutated_results",
                       save_matrices: bool = False, save_pdbs: bool = True) -> dict:
        """
        Complete pipeline: encode complex and generate mutated antibody.

        Args:
            pdb_path: Path to the PDB file
            output_dir: Directory to save results
            save_matrices: Whether to save numpy matrices
            save_pdbs: Whether to save PDB files

        Returns:
            Dictionary with results including original and mutated matrices
        """
        # Encode the complex
        ab_matrix, ag_matrix, pdb_id = self.encode_complex(pdb_path)

        if ab_matrix is None or ag_matrix is None:
            return {"success": False, "error": "Failed to encode complex", "pdb_id": pdb_id}

        # Generate mutated antibody
        try:
            mutated_ab = self.generate_mutated_antibody(ab_matrix, ag_matrix)
            print(f"MUTATED AB res 1 {mutated_ab[0][0]} {pdb_id} {pdb_path}")

            # Create output directory
            makedirs(output_dir, exist_ok=True)
            makedirs(f"{output_dir}/{pdb_id}", exist_ok=True)

            # Save matrices if requested
            if save_matrices:
                np.save(f"{output_dir}/{pdb_id}/original_ab.npy", ab_matrix)
                np.save(f"{output_dir}/{pdb_id}/original_ag.npy", ag_matrix)
                np.save(f"{output_dir}/{pdb_id}/mutated_ab.npy", mutated_ab)

            # Save PDB files if requested
            if save_pdbs:
                # Save mutated complex
                self._create_complex_pdb(
                    mutated_ab, ag_matrix,
                    f"{output_dir}/{pdb_id}/mutated_complex.pdb",
                    pdb_path
                )

                # Save individual chains
                self._decode_matrix_to_pdb(
                    ab_matrix, "H",
                    f"{output_dir}/{pdb_id}/original_antibody.pdb",
                    pdb_path
                )

                self._decode_matrix_to_pdb(
                    mutated_ab, "H",
                    f"{output_dir}/{pdb_id}/mutated_antibody.pdb",
                    pdb_path
                )

                self._decode_matrix_to_pdb(
                    ag_matrix, "A",
                    f"{output_dir}/{pdb_id}/antigen.pdb",
                    pdb_path
                )

            print(f"Mutation completed for {pdb_id}")
            print(f"Results saved to {output_dir}/{pdb_id}/")

            return {
                "success": True,
                "pdb_id": pdb_id,
                "original_ab": ab_matrix,
                "original_ag": ag_matrix,
                "mutated_ab": mutated_ab,
                "output_path": f"{output_dir}/{pdb_id}"
            }

        except Exception as e:
            return {"success": False, "error": f"Mutation failed: {str(e)}", "pdb_id": pdb_id}

    def mutate_all_complexes(self, output_dir: str = "./mutated_results",
                             save_matrices: bool = False, save_pdbs: bool = True):

        for pdb_file in listdir(self.complex_folder):
            if pdb_file.endswith('.pdb'):
                pdb_path = path.join(self.complex_folder, pdb_file)
                print(f"\nProcessing {pdb_file}...")

                result = self.mutate_complex(pdb_path, output_dir, save_matrices, save_pdbs)

                if result["success"]:
                    print(f"Successfully mutated {pdb_file}")
                else:
                    print(f"Failed to mutate {pdb_file}: {result['error']}")


    def quick_test_context_sensitivity(self):
        """Test rapido per verificare context-sensitivity"""
        pdb_files = [f for f in listdir(self.complex_folder) if f.endswith('.pdb')]

        if len(pdb_files) < 2:
            print("Need at least 2 PDB files for testing")
            return

        results = []
        for pdb_file in pdb_files[:2]:  # Test sui primi 2
            pdb_path = path.join(self.complex_folder, pdb_file)
            ab_matrix, ag_matrix, pdb_id = self.encode_complex(pdb_path)

            if ab_matrix is not None and ag_matrix is not None:
                mutated_ab = self.generate_mutated_antibody(ab_matrix, ag_matrix)
                results.append((pdb_id, mutated_ab))

        if len(results) == 2:
            diff = np.mean(np.abs(results[0][1] - results[1][1]))
            print(f"Output difference between {results[0][0]} and {results[1][0]}: {diff:.6f}")

            if diff < 1e-6:
                print("🚨 PROBLEMA: Output identici per input diversi!")
                return False
            else:
                print("✅ OK: Output diversi per input diversi")
                return True