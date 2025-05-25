from keras import layers, Input, Model
import tensorflow as tf

# Constants
AMINO_ACIDS = ["ALA", "ARG", "ASN", "ASP", "CYS", "CYX", "GLN", "GLU", "GLY",
               "HIS", "HIE", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
               "THR", "TRP", "TYR", "VAL"]
ELE_TYPES = ['N.3', "N.am", "N.4", 'C.3', 'C.2', 'O.2', 'O.3', 'O.co2']
MAX_RESIDUES = 92
NUM_ATOMS = 5
XYZQ_DIM = 4  # x, y, z, charge
ATOM_TYPE_DIM = 8
RESNAME_DIM = 22
OUTPUT_DIM = XYZQ_DIM + ATOM_TYPE_DIM + RESNAME_DIM


def Discriminator(ab_shape, ag_shape):
    ab_input = Input(shape=ab_shape, name="ab_input")
    ag_input = Input(shape=ag_shape, name="ag_input")

    def encode_entity(entity):
        entity = layers.TimeDistributed(layers.TimeDistributed(layers.Dense(64, activation='relu')))(entity)
        entity = layers.Reshape((entity.shape[1], -1))(entity)
        entity = layers.Bidirectional(layers.GRU(128, return_sequences=True))(entity)
        entity = layers.GlobalAveragePooling1D()(entity)
        return entity

    x_ab = encode_entity(ab_input)
    x_ag = encode_entity(ag_input)
    x = layers.Concatenate()([x_ab, x_ag])
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    gbsa_out = layers.Dense(1, name="gbsa_prediction")(x)
    validity = layers.Dense(1, activation="sigmoid", name="validity")(x)

    return Model(inputs=[ab_input, ag_input], outputs=[gbsa_out, validity])


# class AutoregressiveGeneratorConditioned(Model):
#     def __init__(self, features_coords=3, charge=1, eleTypes=8, amino_acids=22, embedding_dim=128):
#         super().__init__()
#         self.max_residues = 0
#         self.coords = features_coords
#         self.charge = charge
#         self.eleTypes = eleTypes
#         self.amino_acids = amino_acids
#         self.embedding_dim = embedding_dim
#
#         # Embedding antigene e anticorpo
#         self.ag_embedding = layers.Dense(embedding_dim, activation='relu')
#         self.ab_embedding = layers.Dense(embedding_dim, activation='relu')
#
#         # RNN cell per generare residui sequenzialmente?
#         self.rnn_cell = layers.GRUCell(embedding_dim)
#
#         # Output layers per coordinate, ele, res
#         self.dense_coords = layers.Dense(features_coords)
#         self.dense_charge = layers.Dense(charge, activation="tanh")
#         self.dense_ele = layers.Dense(eleTypes)
#         self.dense_res = layers.Dense(amino_acids)
#
#     def call(self, ag_input, ab_input, training=False):
#         batch_size = tf.shape(ag_input)[0]
#         non_zero_rows = tf.reduce_any(tf.not_equal(ab_input, 0), axis=[2, 3])
#         non_zero_counts = tf.reduce_sum(tf.cast(non_zero_rows, tf.int32), axis=1)
#         self.max_residues = tf.reduce_max(non_zero_counts)
#
#         # Embedding antigene (avg pooling + dense)
#         ag_embed = tf.reduce_mean(ag_input, axis=[1, 2])  # (batch, 34)
#         ag_embed = self.ag_embedding(ag_embed)  # (batch, embedding_dim)
#
#         # Embedding anticorpo (avg pooling + dense)
#         ab_embed = tf.reduce_mean(ab_input, axis=[1, 2])  # (batch, 34)
#         ab_embed = self.ab_embedding(ab_embed)  # (batch, embedding_dim)
#
#         # Stato iniziale RNN
#         state = ag_embed + ab_embed  # (batch, embedding_dim)
#
#         # Input iniziale alla RNN (può essere zero o learnable)
#         input_step = tf.zeros((batch_size, self.embedding_dim))
#
#         coords_all = []
#         num_residues = int(self.max_residues.numpy())
#
#         for r in range(num_residues):  # per ogni residuo
#             coords_atoms = []
#
#             for a in range(NUM_ATOMS):  # per ogni atomo
#                 output, state = self.rnn_cell(input_step, state)
#                 # output, state = self.rnn_cell(input_step, [state])
#
#                 coords = self.dense_coords(output)  # (batch, 4)
#                 charge = self.dense_charge(output)
#                 ele_logits = self.dense_ele(output)  # (batch, 8)
#                 res_logits = self.dense_res(output)  # (batch, 22)
#
#                 ele_prob = tf.nn.softmax(ele_logits, axis=-1)
#                 res_prob = tf.nn.softmax(res_logits, axis=-1)
#
#                 ele_ids = tf.argmax(ele_prob, axis=-1)
#                 res_ids = tf.argmax(res_prob, axis=-1)
#
#                 ele_onehot = tf.one_hot(ele_ids, depth=self.eleTypes, dtype=tf.float32)
#                 res_onehot = tf.one_hot(res_ids, depth=self.amino_acids, dtype=tf.float32)
#
#                 final_atom = tf.concat([coords, charge, ele_onehot, res_onehot], axis=-1)  # (batch, 34)
#
#                 coords_atoms.append(final_atom)
#                 input_step = output  # auto-regressive step
#
#             residue_atoms = tf.stack(coords_atoms, axis=1)  # (batch, NUM_ATOMS, 34)
#             coords_all.append(residue_atoms)
#
#         final_output = tf.stack(coords_all, axis=1)  # (batch, max_residues, NUM_ATOMS, 34)
#         current_shape = tf.shape(final_output)
#         # batch_size, num_residues, num_atoms, num_dims = current_shape[0], current_shape[1], current_shape[2], \
#         # current_shape[3]
#
#         padding_amount = 92 - current_shape[1]
#
#         padded_output = tf.pad(
#             final_output,
#             paddings=[[0, 0], [0, padding_amount], [0, 0], [0, 0]],  # [batch, residues, atoms, dims]
#             mode='CONSTANT',
#             constant_values=0
#         )
#         return padded_output

class AutoregressiveGeneratorConditioned(Model):
    def __init__(self, features_coords=3, charge=1, eleTypes=8, amino_acids=22, embedding_dim=128):
        super().__init__()
        self.coords = features_coords
        self.charge = charge
        self.eleTypes = eleTypes
        self.amino_acids = amino_acids
        self.embedding_dim = embedding_dim

        # Improved embeddings with normalization
        self.ag_embedding = tf.keras.Sequential([
            layers.Dense(embedding_dim, activation='relu'),
            layers.LayerNormalization(),
            layers.Dense(embedding_dim, activation='relu')
        ])
        self.ab_embedding = tf.keras.Sequential([
            layers.Dense(embedding_dim, activation='relu'),
            layers.LayerNormalization(),
            layers.Dense(embedding_dim, activation='relu')
        ])

        # Multi-layer RNN for better sequence modeling
        self.rnn_cell_1 = layers.GRUCell(embedding_dim)
        self.rnn_cell_2 = layers.GRUCell(embedding_dim)

        # Separate specialized heads for different outputs
        self.coords_head = tf.keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.LayerNormalization(),
            layers.Dense(32, activation='relu'),
            layers.Dense(features_coords)  # No activation for coordinates
        ])

        self.charge_head = tf.keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(charge, activation='tanh')  # Bounded between -1 and 1
        ])

        self.ele_head = tf.keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.LayerNormalization(),
            layers.Dense(32, activation='relu'),
            layers.Dense(eleTypes)  # Logits for softmax
        ])

        self.res_head = tf.keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.LayerNormalization(),
            layers.Dense(32, activation='relu'),
            layers.Dense(amino_acids)  # Logits for softmax
        ])

        # Learnable initial state
        self.initial_state_1 = self.add_weight(
            shape=(1, embedding_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='initial_state_1'
        )
        self.initial_state_2 = self.add_weight(
            shape=(1, embedding_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='initial_state_2'
        )

    def call(self, ag_input, ab_input, training=False):
        batch_size = tf.shape(ag_input)[0]

        # Determine actual number of residues from input
        non_zero_rows = tf.reduce_any(tf.not_equal(ab_input, 0), axis=[2, 3])
        non_zero_counts = tf.reduce_sum(tf.cast(non_zero_rows, tf.int32), axis=1)
        max_residues = tf.reduce_max(non_zero_counts)

        # Better embedding approach - use both mean and max pooling
        ag_mean = tf.reduce_mean(ag_input, axis=[1, 2])
        ag_max = tf.reduce_max(ag_input, axis=[1, 2])
        ag_embed = self.ag_embedding(tf.concat([ag_mean, ag_max], axis=-1))

        ab_mean = tf.reduce_mean(ab_input, axis=[1, 2])
        ab_max = tf.reduce_max(ab_input, axis=[1, 2])
        ab_embed = self.ab_embedding(tf.concat([ab_mean, ab_max], axis=-1))

        # Combine conditioning information
        conditioning = ag_embed + ab_embed

        # Initial states for multi-layer RNN
        state_1 = tf.tile(self.initial_state_1, [batch_size, 1]) + conditioning
        state_2 = tf.tile(self.initial_state_2, [batch_size, 1]) + conditioning

        # Simple Gumbel softmax implementation (define once outside loop)
        def gumbel_softmax(logits, temperature):
            gumbel_noise = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(logits), 1e-20, 1.0)))
            return tf.nn.softmax((logits + gumbel_noise) / temperature)

        # Generate up to max_residues for efficiency, then mask appropriately
        coords_all = []

        # Use the calculated max_residues to limit generation
        generation_limit = tf.minimum(max_residues, 92)  # Don't exceed 92

        for r in range(92):  # Fixed range for graph compilation
            # Create mask for which samples should generate this residue
            should_generate = tf.less(r, non_zero_counts)  # Shape: (batch_size,)

            if r >= generation_limit:
                # All remaining residues are zero-padded
                zero_residue = tf.zeros((batch_size, NUM_ATOMS, OUTPUT_DIM))
                coords_all.append(zero_residue)
                continue

            coords_atoms = []

            for a in range(NUM_ATOMS):
                # Multi-layer RNN processing
                output_1, state_1 = self.rnn_cell_1(conditioning, state_1)
                output_2, state_2 = self.rnn_cell_2(output_1, state_2)

                # Generate different components with specialized heads
                coords = self.coords_head(output_2)
                charge = self.charge_head(output_2)
                ele_logits = self.ele_head(output_2)
                res_logits = self.res_head(output_2)

                # Apply temperature scaling for better diversity
                temperature = 0.8 if training else 0.5
                ele_logits = ele_logits / temperature
                res_logits = res_logits / temperature

                if training:
                    ele_onehot = gumbel_softmax(ele_logits, temperature)
                    res_onehot = gumbel_softmax(res_logits, temperature)
                else:
                    # Hard selection during inference
                    ele_prob = tf.nn.softmax(ele_logits, axis=-1)
                    res_prob = tf.nn.softmax(res_logits, axis=-1)
                    ele_ids = tf.argmax(ele_prob, axis=-1)
                    res_ids = tf.argmax(res_prob, axis=-1)
                    ele_onehot = tf.one_hot(ele_ids, depth=self.eleTypes, dtype=tf.float32)
                    res_onehot = tf.one_hot(res_ids, depth=self.amino_acids, dtype=tf.float32)

                # Combine all features
                final_atom = tf.concat([coords, charge, ele_onehot, res_onehot], axis=-1)

                # Apply mask: zero out atoms for samples that shouldn't generate this residue
                mask = tf.expand_dims(tf.cast(should_generate, tf.float32), axis=-1)  # Shape: (batch_size, 1)
                final_atom = final_atom * mask

                coords_atoms.append(final_atom)

                # Use generated atom as input for next atom (autoregressive)
                conditioning = conditioning + tf.reduce_mean(final_atom, axis=-1, keepdims=True) * 0.1

            residue_atoms = tf.stack(coords_atoms, axis=1)  # Shape: (batch_size, 5, 34)
            coords_all.append(residue_atoms)

        final_output = tf.stack(coords_all, axis=1)  # Shape: (batch_size, 92, 5, 34)
        return final_output

# if __name__ == "__main__":
#     import tensorflow as tf
#     from tensorflow.keras import Model, layers
#
#     max_ab_len = 92
#     features_coords = 3
#     eleTypes = 8
#     amino_acids = 22
#     NUM_ATOMS = 5  # Required constant
#
#     # Dummy inputs: [batch, residues, atoms, features]
#     dummy_ag = tf.random.normal((1, 97, NUM_ATOMS, features_coords + eleTypes + amino_acids))
#     dummy_ab = tf.random.normal((1, max_ab_len, NUM_ATOMS, features_coords + eleTypes + amino_acids))
#
#     # Instantiate the model (without max_residues in __init__)
#     model = AutoregressiveGeneratorConditioned()
#
#     # Run a forward pass
#     output = model(dummy_ag, dummy_ab)
#
#     # Check output
#     print("Output shape:", output.shape)  # Should be (1, 92, 5, 34)
#     print("Coords (first atom of second residue):", output[0, 1, 0, :4])  # First 4 = coordinates
#     print("One-hot (ele+res):", output[0, 1, 0, 4:])  # Remaining 30 = one-hot
