import numpy as np

eleTypes = ['N.3', "N.am", "N.4", 'C.3', 'C.2', 'O.2', 'O.3', 'O.co2']
amino_acids = ["ALA", "ARG", "ASN", "ASP", "CYS", "CYX", "GLN", "GLU", "GLY", "HIS", "HIE",
               "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
accepted_atoms = ['N', 'CA', 'CB', 'C', 'O']


def decode_matrix_to_mol2(matrix: np.ndarray, output_path: str, mol2_name="decoded.mol2"):
    atom_lines = []
    atom_id = 1
    res_id = 1

    for residue in matrix:
        # Get most likely residue name
        for i in range(len(amino_acids)):
            residue_onehot_sum = sum(residue[:, 4 + 8 + i])  # index 4 to 4+8 = atom_type, 4+8+i = residue type
            if residue_onehot_sum > 0.5:  # One-hot sum across atoms should be close to 1
                res_name = amino_acids[i]
                break
        else:
            res_name = "UNK"

        for atom_idx, atom_data in enumerate(residue):
            x, y, z = atom_data[0:3]
            charge = atom_data[3]

            # Atom type one-hot
            atom_type_index = np.argmax(atom_data[4:12])
            atom_type = eleTypes[atom_type_index]

            atom_name = accepted_atoms[atom_idx]

            line = f"{atom_id:7d} {atom_name:<4} {x:9.4f} {y:9.4f} {z:9.4f} {atom_type:<5} {res_name}{res_id:>4} {res_name} {charge: .4f}"
            atom_lines.append(line)
            atom_id += 1
        res_id += 1

    mol2_content = "@<TRIPOS>MOLECULE\n"
    mol2_content += f"{mol2_name}\n"
    mol2_content += f"{len(atom_lines)} 0 0 0 0\n"
    mol2_content += "SMALL\nNO_CHARGES\n\n"

    mol2_content += "@<TRIPOS>ATOM\n"
    mol2_content += "\n".join(atom_lines)
    mol2_content += "\n@<TRIPOS>BOND\n"

    with open(output_path, "w") as f:
        f.write(mol2_content)

    print(f"Written {mol2_name} to {output_path}")
