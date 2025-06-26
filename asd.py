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

# genetic
#
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


# amino_acids = ["ALA", "ARG", "ASN", "ASP", "CYS", "CYX", "GLN", "GLU", "GLY", "HIS", "HIE",
#                        "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
#
# print(len(amino_acids))

# from tensorflow.keras import layers, Input, Model
# import numpy as np
# from Model.Models import Generator
# import sys
# np.set_printoptions(threshold=sys.maxsize)
#
# accepted_atoms = ['N', 'CA', 'CB', 'C', 'O']
# resnames = ["ALA", "ARG", "ASN", "ASP", "CYS", "CYX", "GLN", "GLU", "GLY", "HIS", "HIE",
#             "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
#
# atom_names_one_hot = np.eye(len(accepted_atoms))
# resnames_one_hot = np.eye(len(resnames))
#
# max_ab_len = 92
# max_ag_len = 97
#
# antigen_residues = 34
# antibody_residues = 16
# number_of_atoms = 5
#
#
# def GenerateSample(number_of_residues: int, what: str):
#     if what == "ag":
#         sample = np.zeros(shape=(max_ag_len, 5, 30))
#     else:
#         sample = np.zeros(shape=(max_ab_len, 5, 30))
#     for x in range(0, number_of_residues):
#         amminoacido_scelto = np.random.choice(len(resnames))
#         for atomo in range(len(accepted_atoms)):
#             if resnames[amminoacido_scelto] == "GLY" and atomo == 2:
#                 continue
#             coordinates = np.random.uniform(size=(3,), low=-100, high=180).round(2)
#             zio = np.concatenate((coordinates, atom_names_one_hot[atomo], resnames_one_hot[amminoacido_scelto]), axis=0)
#             sample[x][atomo] = zio
#     return sample
#
#
# ag = GenerateSample(48, 'ag')
# ab = GenerateSample(42, 'ab')
#
# ab_sample = np.expand_dims(ab, axis=0)
# ag_sample = np.expand_dims(ag, axis=0)
# # print(ab.shape, ag.shape) sono già padded! ottimo 92, 5, 30 - 97, 5, 30
# # print(ag_sample.shape, ab_sample.shape)    1, 97, 5, 30... etc
#
# ab_test = np.random.rand(1, 92, 5, 30).astype(np.float32)
# ag_test = np.random.rand(1, 97, 5, 30).astype(np.float32)
#
#
# generator = Generator()
# generator.summary()
# print(ab_sample[0][0])
# print("TYPES:", ab_sample.dtype, ab_sample.dtype)
# out = generator.predict([ab_test, ag_test])
# for x in out[0][0]:
#     print(x)

# discriminator = load_model("./best_model.keras", compile=False, safe_mode=False)
# discriminator.trainable = False
#
# ab_ottimizzato, gbsa_finale = MutagenesiGuidata(
#     discriminatore=discriminator,
#     ab=ab,
#     ag=ag,
#     resnames_onehot=resnames_one_hot,
# )
