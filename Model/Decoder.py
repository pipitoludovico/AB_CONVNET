import numpy as np
import os

eleTypes = ['N.3', "N.am", "N.4", 'C.3', 'C.2', 'O.2', 'O.3', 'O.co2']
amino_acids = ["ALA", "ARG", "ASN", "ASP", "CYS", "CYX", "GLN", "GLU", "GLY", "HIS", "HIE",
               "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
accepted_atoms = ['N', 'CA', 'CB', 'C', 'O']


def decode_matrix_to_mol2(matrix: np.ndarray, output_dir: str, mol2_prefix="decoded"):
    """
    Decode a batch of matrices into individual mol2 files.
    """
    print("Matrix shape:", matrix.shape)
    os.makedirs(output_dir, exist_ok=True)

    num_samples = matrix.shape[0]

    for sample_idx in range(num_samples):
        sample = matrix[sample_idx]  # shape (97, 5, 34)
        print(sample)
        atom_lines = []
        atom_id = 1
        res_id = 1

        for residue in sample:
            # Determine residue name
            for i, aa in enumerate(amino_acids):
                # Check if this residue was labeled as amino_acid[i]
                residue_onehot_sum = np.sum(residue[:, 12 + i])  # residue type starts at index 12
                if residue_onehot_sum > 0.5:
                    res_name = aa
                    break
            else:
                res_name = "UNK"

            for atom_idx, atom_data in enumerate(residue):
                x, y, z = atom_data[0:3]
                charge = atom_data[3]

                atom_type_index = np.argmax(atom_data[4:12])
                atom_type = eleTypes[atom_type_index]

                atom_name = accepted_atoms[atom_idx]

                line = f"{atom_id:7d} {atom_name:<4} {x:9.4f} {y:9.4f} {z:9.4f} {atom_type:<5} {res_name}{res_id:>4} {res_name} {charge: .4f}"
                atom_lines.append(line)
                atom_id += 1
            res_id += 1

        mol2_content = "@<TRIPOS>MOLECULE\n"
        mol2_content += f"{mol2_prefix}_{sample_idx}\n"
        mol2_content += f"{len(atom_lines)} 0 0 0 0\n"
        mol2_content += "SMALL\nNO_CHARGES\n\n"
        mol2_content += "@<TRIPOS>ATOM\n"
        mol2_content += "\n".join(atom_lines)
        mol2_content += "\n@<TRIPOS>BOND\n"

        file_path = os.path.join(output_dir, f"{mol2_prefix}_{sample_idx}.mol2")
        with open(file_path, "w") as f:
            f.write(mol2_content)
        print(f"Written {file_path}")


# Load and decode
ag = np.load('../generated_ag.npy')  # shape (10, 97, 5, 34)
decode_matrix_to_mol2(ag, output_dir="../decoded_mol2s", mol2_prefix="ab_generated")
