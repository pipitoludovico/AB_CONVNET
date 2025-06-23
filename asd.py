# import tensorflow as tf
# from tensorflow.keras import layers, Model
#
#
# class AutoregressiveGeneratorConditioned(Model):
#     def __init__(self, max_residues=92, features_coords=4, eleTypes=8, amino_acids=22, embedding_dim=128):
#         super().__init__()
#         self.max_residues = max_residues
#         self.features_coords = features_coords
#         self.eleTypes = eleTypes
#         self.amino_acids = amino_acids
#         self.embedding_dim = embedding_dim
#
#         # Embedding antigene e anticorpo
#         self.ag_embedding = layers.Dense(embedding_dim, activation='relu')
#         self.ab_embedding = layers.Dense(embedding_dim, activation='relu')
#
#         # RNN cell per generare residui sequenzialmente
#         self.rnn_cell = layers.GRUCell(embedding_dim)
#
#         # Output layers per coordinate, ele, res
#         self.dense_coords = layers.Dense(features_coords)
#         self.dense_ele = layers.Dense(eleTypes)
#         self.dense_res = layers.Dense(amino_acids)
#
#     def call(self, ag_input, ab_input, training=False):
#         batch_size = tf.shape(ag_input)[0]
#
#         # Embedding antigene (avg pooling + dense)
#         ag_embed = tf.reduce_mean(ag_input, axis=[1, 2])
#         ag_embed = self.ag_embedding(ag_embed)
#
#         # Embedding anticorpo (avg pooling + dense)
#         ab_embed = tf.reduce_mean(ab_input, axis=1)
#         ab_embed = self.ab_embedding(ab_embed)
#
#         # Combina embedding (es. somma o concatenazione)
#         combined_embed = ag_embed + ab_embed  # o tf.concat([ag_embed, ab_embed], axis=-1) e modifica stato RNN
#
#         # Stato iniziale RNN
#         state = combined_embed
#
#         coords_seq = []
#         ele_seq = []
#         res_seq = []
#
#         input_step = tf.zeros((batch_size, self.embedding_dim))
#
#         for t in range(self.max_residues):
#             output, state = self.rnn_cell(input_step, state)
#
#             coords = self.dense_coords(output)
#             ele_logits = self.dense_ele(output)
#             res_logits = self.dense_res(output)
#
#             coords_seq.append(coords)
#             ele_seq.append(ele_logits)
#             res_seq.append(res_logits)
#
#             input_step = output
#
#         coords_seq = tf.stack(coords_seq, axis=1)
#         ele_seq = tf.stack(ele_seq, axis=1)
#         res_seq = tf.stack(res_seq, axis=1)
#
#         ele_prob = tf.nn.softmax(ele_seq, axis=-1)
#         res_prob = tf.nn.softmax(res_seq, axis=-1)
#
#         ele_ids = tf.argmax(ele_prob, axis=-1)
#         res_ids = tf.argmax(res_prob, axis=-1)
#
#         ele_onehot = tf.one_hot(ele_ids, depth=self.eleTypes, dtype=tf.float32)
#         res_onehot = tf.one_hot(res_ids, depth=self.amino_acids, dtype=tf.float32)
#
#         final_output = tf.concat([coords_seq, ele_onehot, res_onehot], axis=-1)
#         return final_output
#
#
# if __name__ == "__main__":
#     max_ab_len = 92
#     features_coords = 4
#     eleTypes = 8
#     amino_acids = 22
#
#     dummy_ag = tf.random.normal((1, 97, 5, features_coords + eleTypes + amino_acids))
#     dummy_ab = tf.random.normal((1, max_ab_len, features_coords + eleTypes + amino_acids))
#
#     model = AutoregressiveGeneratorConditioned(max_residues=max_ab_len,
#                                                features_coords=features_coords,
#                                                eleTypes=eleTypes,
#                                                amino_acids=amino_acids)
#
#     output = model(dummy_ag, dummy_ab)
#
#     print(output.shape)
#     print(output[0, 1, :4])  # prime 4 features coordinate del primo residuo
#     print(output[0, 1, 4:])  # features one-hot (ele + res) del primo residuo

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


# import numpy as np
#
# # Example data with shape (12, 30, 5, 34)
# data = np.random.randn(12, 30, 5, 34)
#
# print(f"Data shape: {data.shape}")
# # 1. Pad the matrix from (12, 30, 5, 34) to (12, 92, 5, 34)
# padded_data = np.pad(data,
#                      pad_width=((0, 0), (0, 92 - 30), (0, 0), (0, 0)),
#                      mode='constant',
#                      constant_values=0)
#
# print(f"Padded shape: {padded_data.shape}")  # Should be (12, 92, 5, 34)
#
# # 2. Count non-zero rows in original data (for each sample in batch)
# # Method 1: Vectorized approach (fastest)
# non_zero_counts = np.sum(np.any(data != 0, axis=(2, 3)), axis=1)
#
# print(f"Non-zero rows per sample (vectorized): {non_zero_counts}")

amino_acids = ["ALA", "ARG", "ASN", "ASP", "CYS", "CYX", "GLN", "GLU", "GLY", "HIS", "HIE",
                       "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]

print(len(amino_acids))