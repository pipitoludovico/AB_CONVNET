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


import numpy as np

# Example data with shape (12, 30, 5, 34)
data = np.random.randn(12, 30, 5, 34)

print(f"Data shape: {data.shape}")
# 1. Pad the matrix from (12, 30, 5, 34) to (12, 92, 5, 34)
padded_data = np.pad(data,
                    pad_width=((0, 0), (0, 92-30), (0, 0), (0, 0)),
                    mode='constant',
                    constant_values=0)

print(f"Padded shape: {padded_data.shape}")  # Should be (12, 92, 5, 34)

# 2. Count non-zero rows in original data (for each sample in batch)
# Method 1: Vectorized approach (fastest)
non_zero_counts = np.sum(np.any(data != 0, axis=(2, 3)), axis=1)

print(f"Non-zero rows per sample (vectorized): {non_zero_counts}")
