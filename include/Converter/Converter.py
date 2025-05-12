import numpy as np
import MDAnalysis as Mda


def features_to_mol2(generated_features, eleTypes, amino_acids,
                     template_mol2='complex_minimized.mol2',
                     output_path='generated.mol2'):
    feats = generated_features.flatten()
    num_atoms = feats.shape[0] // 32
    atom_feats = feats[:num_atoms * 32].reshape(num_atoms, 32)

    # Load template so we can reuse its bonds & atom/residue ordering
    u = Mda.Universe(template_mol2)

    with open(output_path, 'w') as fout:
        # Copy molecule header (everything up to the first @<TRIPOS>ATOM)
        header, rest = open(template_mol2).read().split('@<TRIPOS>ATOM', 1)
        fout.write(header)
        fout.write('@<TRIPOS>ATOM\n')

        # Overwrite just the coordinates / charges / types
        for i, atom in enumerate(atom_feats, start=1):
            x, y, z = atom[0], atom[1], atom[2]
            charge = atom[3]
            atype = eleTypes[np.argmax(atom[4:4 + len(eleTypes)])]
            residx = u.atoms[i - 1].resid
            resname = amino_acids[np.argmax(atom[4 + len(eleTypes):])]
            fout.write(f'{i:>7d} {u.atoms[i - 1].name:<4s} {x:8.3f} {y:8.3f} {z:8.3f} '
                       f'{atype:<6s} {residx:>3d} {resname:<4s} {charge:8.4f}\n')

        # Now append the bond section, taken directly from the template:
        fout.write('\n@<TRIPOS>BOND\n')
        for j, bond in enumerate(u.bonds, start=1):
            a1 = bond.atoms[0].ix + 1  # Mdanalysis is zero-based
            a2 = bond.atoms[1].ix + 1
            fout.write(f'{j:>6d} {a1:>4d} {a2:>4d} 1\n')
