from keras import layers, Input, Model
import numpy as np
from .Custom_Layers import (
    CoordinateExtractor, AtomExtractor, CoordinateSum, PaddingMask,
    MaskSqueezer, LogitsMasker, GumbelSoftmax, ResidueExpander,
    ResidueTiler, MaskTiler
)


def Discriminator(atoms_per_res=5, feature_dim=30):
    ab_input = Input(shape=(92, atoms_per_res, feature_dim), name='ab_input')
    ag_input = Input(shape=(97, atoms_per_res, feature_dim), name='ag_input')

    def per_atom_embedding_split(vector):
        # Split the 30 features into meaningful components
        coords = vector[..., :3]  # (batch, residues, atoms, 3) - x,y,z coordinates
        atom_types = vector[..., 3:8]  # (batch, residues, atoms, 5) - atom type sparse vector
        residue_types = vector[..., 8:30]  # (batch, residues, atoms, 22) - residue type sparse vector

        # Process coordinates with spatial awareness
        coords_proj = layers.Dense(32, activation='relu')(coords)
        coords_proj = layers.Dense(64, activation='relu')(coords_proj)

        # Process atom types (categorical)
        atom_proj = layers.Dense(32, activation='relu')(atom_types)
        atom_proj = layers.Dense(32, activation='relu')(atom_proj)

        # Process residue types (categorical)
        residue_proj = layers.Dense(32, activation='relu')(residue_types)
        residue_proj = layers.Dense(32, activation='relu')(residue_proj)

        # Combine all representations
        combined_ = layers.Concatenate()([coords_proj, atom_proj, residue_proj])  # (batch, residues, atoms, 128)

        return combined_

    def process_chain(chain_input, n_res):
        x_ = per_atom_embedding_split(chain_input)  # (batch, residues, atoms, 128)
        x_ = layers.Reshape((n_res, atoms_per_res * 128))(x_)  # (batch, residues, 640)

        # Per-residue processing
        x_ = layers.Conv1D(256, kernel_size=3, padding='same', activation='relu')(x_)
        x_ = layers.BatchNormalization()(x_)
        x_ = layers.Conv1D(256, kernel_size=3, padding='same', activation='relu')(x_)
        x_ = layers.BatchNormalization()(x_)
        x_ = layers.GlobalAveragePooling1D()(x_)
        return x_  # (batch, 256)

    ab_repr = process_chain(ab_input, 92)
    ag_repr = process_chain(ag_input, 97)

    combined = layers.Concatenate()([ab_repr, ag_repr])  # (batch, 512)
    x = layers.Dense(512, activation='relu')(combined)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)

    gbsa_output = layers.Dense(1, name='gbsa_prediction')(x)
    validity_output = layers.Dense(1, activation='sigmoid', name='validity')(x)

    return Model(inputs=[ab_input, ag_input], outputs=[gbsa_output, validity_output])


def Generator(atoms_per_res=5, feature_dim=30, ab_max_len=92, ag_max_len=97, temperature=1.0, hard_gumbel=False):
    """
    Generator that learns to mutate antibody residue types to improve GBSA.
    Handles 0-padded sequences using Masking layers.
    Uses Gumbel-Softmax for differentiable categorical sampling.
    """

    ab_input = Input(shape=(ab_max_len, atoms_per_res, feature_dim), name='ab_input')
    ag_input = Input(shape=(ag_max_len, atoms_per_res, feature_dim), name='ag_input')

    # Flatten inputs for masking
    ab_flat = layers.Reshape((-1,))(ab_input)  # (batch, ab_len * atoms * features)
    ag_flat = layers.Reshape((-1,))(ag_input)  # (batch, ag_len * atoms * features)

    # Apply masking to handle 0-padding
    ab_masked = layers.Masking(mask_value=0.0, name='ab_masking')(ab_flat)
    ag_masked = layers.Masking(mask_value=0.0, name='ag_masking')(ag_flat)

    # --- ANTIGEN PROCESSING FOR CONDITIONING ---
    ag_context = layers.Dense(512, activation='relu', name='ag_encoder_1')(ag_masked)
    ag_context = layers.Dense(256, activation='relu', name='ag_encoder_2')(ag_context)
    ag_context = layers.Dense(128, activation='relu', name='ag_context')(ag_context)

    # --- ANTIBODY PROCESSING ---
    ab_context = layers.Dense(512, activation='relu', name='ab_encoder_1')(ab_masked)
    ab_context = layers.Dense(256, activation='relu', name='ab_encoder_2')(ab_context)
    ab_context = layers.Dense(128, activation='relu', name='ab_context')(ab_context)

    # --- MUTATION NETWORK ---
    # Combine antibody and antigen contexts
    combined_context = layers.Concatenate(name='combined_context')([ab_context, ag_context])

    # Learn mutation patterns
    mutation_net = layers.Dense(256, activation='relu', name='mutation_1')(combined_context)
    mutation_net = layers.Dropout(0.3)(mutation_net)
    mutation_net = layers.Dense(512, activation='relu', name='mutation_2')(mutation_net)
    mutation_net = layers.Dropout(0.3)(mutation_net)

    # Generate logits for residue types
    total_residue_features = ab_max_len * 22  # Only per-residue, not per-atom
    residue_logits_flat = layers.Dense(total_residue_features, name='residue_logits')(mutation_net)

    # Reshape to (batch, residues, 22)
    residue_logits = layers.Reshape((ab_max_len, 22))(residue_logits_flat)

    # Separate the original features using custom layers
    ab_coords = CoordinateExtractor()(ab_input)  # (batch, ab_len, atoms, 3)
    ab_atoms = AtomExtractor()(ab_input)  # (batch, ab_len, atoms, 5)

    # Create mask for padded positions based on coordinates
    coord_sum = CoordinateSum()(ab_coords)

    padding_mask = PaddingMask()(coord_sum)

    # Squeeze mask to (batch, residues, 1) for broadcasting with logits
    padding_mask_2d = MaskSqueezer()(padding_mask)

    # Apply mask to logits (set padded positions to very negative values)
    masked_logits = LogitsMasker()([residue_logits, padding_mask_2d])

    # Apply Gumbel-Softmax
    residue_probs = GumbelSoftmax(temperature=temperature, hard=hard_gumbel)(masked_logits)

    # Expand to all atoms in each residue (consistent across atoms)
    mutated_residues_per_atom = ResidueExpander()(residue_probs)

    mutated_residues = ResidueTiler(atoms_per_res=atoms_per_res)(mutated_residues_per_atom)

    # Expand mask to match residue features shape
    residue_mask = MaskTiler(atoms_per_res=atoms_per_res, num_residue_types=22)(padding_mask)

    # Apply mask to residues (zero out padded positions)
    mutated_residues_masked = layers.Multiply()([mutated_residues, residue_mask])

    # Reconstruct the full antibody structure
    mutated_ab = layers.Concatenate(axis=-1)([ab_coords, ab_atoms, mutated_residues_masked])

    return Model(inputs=[ab_input, ag_input], outputs=mutated_ab)


def GeneticGenerator(ab: np.ndarray, ag: np.ndarray, residue_onehot: np.ndarray,
                     discriminator, max_len=92, verbose=True):
    """
    Perform guided genetic mutation of antibody residues to minimize a discriminator's score.

    Parameters:
        ab (np.ndarray): Antibody input array of shape (1, max_len, 5, 1, 30)
        ag (np.ndarray): Antigen input array of shape (1, ..., ..., ..., 30)
        residue_onehot (np.ndarray): Identity matrix or one-hot array of size (22, 22)
        discriminator (keras.Model): The discriminator model returning a scalar
        max_len (int): Max length of antibody residues
        verbose (bool): Whether to print progress

    Returns:
        Tuple[np.ndarray, float]: Mutated antibody and final discriminator score
    """

    def extract_valid_residues(ab_batch):
        ab_clean = ab_batch[0]  # (92, 5, 1, 30)
        mask = np.any(ab_clean != 0, axis=(1, 3)).squeeze()
        return ab_clean[mask], mask

    def mutate_residue(ab_valid_):
        if ab_valid_.shape[0] == 0:
            raise ValueError("No valid residues found for mutation.")

        idx = np.random.randint(ab_valid_.shape[0])
        old_onehot = ab_valid_[idx][0, 0, -22:]
        old_idx = np.argmax(old_onehot)

        possible_idxs = list(set(range(residue_onehot.shape[0])) - {old_idx})
        new_idx = np.random.choice(possible_idxs)
        new_onehot = residue_onehot[new_idx]

        for atom in range(5):
            ab_valid_[idx][atom, 0, -22:] = new_onehot

        return ab_valid_

    def repad_to_full_length(ab_stripped, max_len_):
        padded = np.zeros((1, max_len_, 5, 1, 30), dtype=ab_stripped.dtype)
        padded[0, :ab_stripped.shape[0]] = ab_stripped
        return padded

    ab_current = ab.copy()
    ab_valid, valid_mask = extract_valid_residues(ab_current)
    n_valid = np.sum(valid_mask)

    if verbose:
        print(f"[INFO] Valid residues before mutation: {n_valid}")

    best_score = discriminator.predict([ab_current, ag])[0].item()
    if verbose:
        print(f"[INFO] Initial GBSA score: {best_score:.4f}")

    for i in range(n_valid):
        ab_mutated_valid = mutate_residue(ab_valid.copy())
        ab_mutated = repad_to_full_length(ab_mutated_valid, max_len_=max_len)

        score = discriminator.predict([ab_mutated, ag])[0].item()
        if verbose:
            print(f"[Mut {i + 1}] Score: {score:.4f} (Best: {best_score:.4f})")

        if score < best_score:
            if verbose:
                print(f"→ Accepted (Improved: {best_score:.4f} → {score:.4f})")
            ab_current = ab_mutated
            ab_valid = ab_mutated[0][valid_mask]  # update working version
            best_score = score
        elif verbose:
            print("→ Rejected")

    return ab_current, best_score

# def GeneticGenerator(ab: np.ndarray, ag: np.ndarray, residue_onehot):
#     def GeneticMutate(ab: np.ndarray, resnames_onehot):
#         # ab shape: (1, 92, 5, 1, 30)
#         ab_clean = ab[0]  # (92, 5, 1, 30)
#
#         mask = np.any(ab_clean != 0, axis=(1, 3)).squeeze()  # shape (92,)
#
#         ab_stripped = ab_clean[mask]  # shape (num_valid_residues, 5, 1, 30)
#         if ab_stripped.shape[0] == 0:
#             raise ValueError("Nessun residuo valido trovato per mutazione.")
#
#         idx_residuo = np.random.randint(ab_stripped.shape[0])
#
#         num_resnames = resnames_onehot.shape[0]
#         eye = np.eye(num_resnames)
#
#         old_onehot = ab_stripped[idx_residuo][0, 0, -22:]
#         old_idx = np.argmax(old_onehot)
#
#         possibili = list(set(range(num_resnames)) - {old_idx})
#         nuovo_idx = np.random.choice(possibili)
#         nuovo_onehot = eye[nuovo_idx]
#
#         for atom in range(5):
#             ab_stripped[idx_residuo][atom, 0, -22:] = nuovo_onehot
#
#         print(f"shape dello strippato: {ab_stripped.shape}")
#         return ab_stripped
#
#     def pad_ab(ab_stripped, max_len=92):
#         padded = np.zeros((1, max_len, 5, 1, 30), dtype=ab_stripped.dtype)
#         n = ab_stripped.shape[0]
#         padded[0, :n] = ab_stripped
#         return padded
#
#     def MutagenesiGuidata(discriminatore, ab, ag, resnames_onehot):
#         ab_current = ab.copy()
#         initial_mask = np.any(ab_current != 0, axis=(2, 3, 4))[0]
#         initial_valid_residues = np.sum(initial_mask)
#         print(f"Residui validi: {initial_valid_residues}")
#
#         gbsa_attuale = discriminatore.predict([ab_current, ag])[0]
#         gbsa_attuale_scalar = gbsa_attuale.item()
#         print(f"GBSA INIZIALE: {gbsa_attuale_scalar:.4f}")
#
#         for i in range(initial_valid_residues):
#             ab_clean = ab_current[0]
#             mask = np.any(ab_clean != 0, axis=(1, 3)).squeeze()
#             ab_stripped = ab_clean[mask]
#
#             ab_mutato_stripped = GeneticMutate(np.expand_dims(ab_stripped, axis=0),
#                                                resnames_onehot)  # (num_valid,5,1,30)
#
#             # Riaggiungo padding e batch
#             ab_mutato = pad_ab(ab_mutato_stripped, max_len=ab_current.shape[1])
#
#             gbsa_mutato = discriminatore.predict([ab_mutato, ag])[0]
#             gbsa_mutato_scalar = gbsa_mutato.item()
#             print(f"[Mutazione {i + 1}] GBSA mutato: {gbsa_mutato_scalar:.4f}")
#
#             if gbsa_mutato_scalar < gbsa_attuale_scalar:
#                 print(f"Mutazione accettata. Miglioramento: {gbsa_attuale_scalar:.4f} → {gbsa_mutato_scalar:.4f}")
#                 ab_current = ab_mutato
#                 gbsa_attuale_scalar = gbsa_mutato_scalar
#             else:
#                 print("Mutazione scartata.")
#
#         print("CURRENT SHAPE", ab_current.shape)
#         return ab_current, gbsa_attuale_scalar
#
#     print(ab.shape, ag.shape, residue_onehot)
