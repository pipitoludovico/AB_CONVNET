from keras import layers, Input, Model
import numpy as np


def Discriminator(ab_shape, ag_shape):
    def conv3d_block(input_tensor):
        x_ = layers.Conv3D(32, (3, 3, 3), padding='same', activation='relu')(input_tensor)
        x_ = layers.BatchNormalization()(x_)
        x_ = layers.MaxPool3D(pool_size=(2, 2, 1))(x_)

        x_ = layers.Conv3D(64, (3, 3, 3), padding='same', activation='relu')(x_)
        x_ = layers.BatchNormalization()(x_)
        x_ = layers.MaxPool3D(pool_size=(2, 2, 1))(x_)

        x_ = layers.Conv3D(128, (3, 3, 3), padding='same', activation='relu')(x_)
        x_ = layers.GlobalAveragePooling3D()(x_)
        return x_

    ab_input = Input(shape=ab_shape, name='ab_input')
    ag_input = Input(shape=ag_shape, name='ag_input')

    x_ab = conv3d_block(ab_input)
    x_ag = conv3d_block(ag_input)

    x = layers.Concatenate()([x_ab, x_ag])
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.1)(x)

    gbsa_out = layers.Dense(1, name='gbsa_prediction')(x)
    validity = layers.Dense(1, activation='sigmoid', name='validity')(x)

    return Model(inputs=[ab_input, ag_input], outputs=[gbsa_out, validity])


def Generator(latent_dim, ag_shape, ab_shape):
    # Remove batch dimension from shapes for Input layer
    ag_input_shape = ag_shape[1:] if len(ag_shape) == 4 else ag_shape
    ab_output_shape = ab_shape[1:] if len(ab_shape) == 4 else ab_shape

    # Add channel dimension if not present (for Conv3D)
    if len(ag_input_shape) == 3:
        ag_input_shape = ag_input_shape + (1,)  # (97, 5, 30, 1)

    ag_input = Input(shape=ag_input_shape, name="antigen_input")
    z_input = Input(shape=(latent_dim,), name="noise_input")

    # Antigen encoder
    x_ag = layers.Conv3D(32, (3, 3, 3), padding="same", activation="relu")(ag_input)
    x_ag = layers.BatchNormalization()(x_ag)
    x_ag = layers.MaxPool3D(pool_size=(2, 2, 2))(x_ag)
    x_ag = layers.Conv3D(64, (3, 3, 3), padding="same", activation="relu")(x_ag)
    x_ag = layers.GlobalAveragePooling3D()(x_ag)

    # Combine with latent vector
    x = layers.Concatenate()([x_ag, z_input])
    x = layers.Dense(256, activation="relu")(x)

    # Final dense to output full ab tensor
    total_ab_elements = np.prod(ab_output_shape)
    x = layers.Dense(total_ab_elements, activation="linear")(x)
    x = layers.Reshape(ab_output_shape)(x)  # Reshape to (92, 5, 30)

    # Decompose into components
    coords = layers.Lambda(lambda x: x[..., :3], name="coords")(x)
    atoms_logits = layers.Lambda(lambda x: x[..., 3:8], name="atoms_logits")(x)
    residues_logits = layers.Lambda(lambda x: x[..., 8:], name="residues_logits")(x)

    atoms_one_hot = layers.Softmax(axis=-1, name="atoms_one_hot")(atoms_logits)
    residues_one_hot = layers.Softmax(axis=-1, name="residues_one_hot")(residues_logits)

    # Combine outputs
    output = layers.Concatenate(axis=-1, name="generated_antibody")([coords, atoms_one_hot, residues_one_hot])

    return Model(inputs=[ag_input, z_input], outputs=output, name="generator")


def GeneticGenerator(ab: np.ndarray, ag: np.ndarray, residue_onehot):
    ab = ab[0]
    ag = ag[0]
    print(ab.shape, ag.shape)

