from keras import layers

from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import Input
import tensorflow as tf
from .Custom_Layers import Separator, DiversityCalculator, ABAgInteractionLayer, PositionalAttentionPooling, \
    ResiduePositionMask


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
    diversity_value = DiversityCalculator(name='diversity_calculator')(ab_input)

    return Model(inputs=[ab_input, ag_input], outputs=[gbsa_output, diversity_value])


class MaskedGlobalAveragePooling2D(layers.Layer):
    """Global average pooling that ignores padded (zero) positions."""

    def __init__(self, **kwargs):
        super(MaskedGlobalAveragePooling2D, self).__init__(**kwargs)

    def call(self, inputs):
        # inputs shape: (batch, height, width, channels)
        # Create mask: non-zero positions
        mask = tf.reduce_sum(tf.abs(inputs), axis=-1, keepdims=True)  # (batch, h, w, 1)
        mask = tf.cast(mask > 0, tf.float32)

        # Apply mask
        masked_inputs = inputs * mask

        # Compute masked average
        sum_vals = tf.reduce_sum(masked_inputs, axis=[1, 2])  # (batch, channels)
        count_vals = tf.reduce_sum(mask, axis=[1, 2]) + 1e-8  # (batch, 1)

        return sum_vals / count_vals

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


def Generator(atoms_per_res=5, feature_dim=30, ab_max_len=92, ag_max_len=97, n_sparse_mutations=22):
    ab_input = Input(shape=(ab_max_len, atoms_per_res, feature_dim), name='ab_input')
    ag_input = Input(shape=(ag_max_len, atoms_per_res, feature_dim), name='ag_input')

    # Separate conserved parts (coordinates + atom type)
    conserved_parts = Separator()(ab_input)  # Shape: (batch, 92, 5, 8)

    # **Key Enhancement: Model AB-AG interactions first**
    ab_with_ag_context = ABAgInteractionLayer(hidden_dim=64)([ab_input, ag_input])

    # Feature extraction with interaction-aware AB and original AG
    ab_conv = layers.Conv2D(128, (3, 1), activation='relu', padding='same')(ab_with_ag_context)
    ab_features = PositionalAttentionPooling(hidden_dim=128)(ab_conv)

    ag_conv = layers.Conv2D(128, (3, 1), activation='relu', padding='same')(ag_input)
    ag_features = PositionalAttentionPooling(hidden_dim=128)(ag_conv)

    # Combined context with interaction information
    context = layers.Concatenate()([ab_features, ag_features])
    context = layers.Dense(256, activation='relu')(context)
    context = layers.Dropout(0.1)(context)  # Add some regularization
    context = layers.Dense(256, activation='relu')(context)

    # Generate mutation probabilities per residue position
    mutation_logits = layers.Dense(ab_max_len * n_sparse_mutations, activation='linear')(context)
    mutation_probs = layers.Reshape((ab_max_len, 1, n_sparse_mutations))(mutation_logits)
    mutation_probs = layers.Softmax(axis=-1)(mutation_probs)

    # Create position mask for AB sequence (residue-level)
    ab_residue_mask = ResiduePositionMask()(ab_input)

    # Apply mask to mutation probabilities (zero out padded positions)
    mutation_probs_masked = layers.Multiply()([mutation_probs, ab_residue_mask])

    # Broadcast to all atoms in each residue using UpSampling2D
    mutation_features = layers.UpSampling2D(size=(1, atoms_per_res))(mutation_probs_masked)
    # Shape: (batch, ab_max_len, atoms_per_res, n_sparse_mutations)

    # Combine conserved (8) + mutation (22) = 30 features total
    mutated_ab = layers.Concatenate(axis=-1)([conserved_parts, mutation_features])
    # Shape: (batch, ab_max_len, atoms_per_res, 30) = (batch, 92, 5, 30)

    # Diversity calculation
    diversity_score = DiversityCalculator()(mutated_ab)

    return Model(inputs=[ab_input, ag_input], outputs=[mutated_ab, diversity_score], name='Generator')
