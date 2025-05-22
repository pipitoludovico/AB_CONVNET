import tensorflow as tf
from keras import layers, Model, Input


class CNNConditionalGenerator(Model):
    def __init__(self, number_of_residues: int = 5, feature_dim=34, atom_num=5, hidden_dim=256):
        super().__init__()

        # Encode antigene con CNN + GRU simile discriminator per coerenza
        self.conv2D = layers.TimeDistributed(layers)
        self.conv1 = layers.TimeDistributed(layers.Conv1D(64, 3, padding='same', activation='relu'))
        self.gru_ag = layers.Bidirectional(layers.GRU(hidden_dim, return_sequences=False))
        self.dense_emb = layers.Dense(hidden_dim, activation='relu')

        # Embedding per posizione residui anticorpo
        self.pos_embedding = layers.Embedding(input_dim=max_residues, output_dim=hidden_dim)

        # Decoder RNN + dense per generare coordinate backbone + features
        self.gru_dec = layers.GRU(hidden_dim, return_sequences=True)
        self.dense_out = layers.TimeDistributed(layers.Dense(atom_num * feature_dim))  # output shape (B, max_res, 5*34)

    def call(self, inputs):
        ag, K = inputs  # ag shape (B, residues, atoms, features)
        B = tf.shape(ag)[0]

        batch_size = tf.shape(ag)[0]
        residues = tf.shape(ag)[1]
        atoms = tf.shape(ag)[2]
        features = tf.shape(ag)[3]

        ag_flat = tf.reshape(ag, [batch_size, residues, atoms * features])  # (B, residues, atoms*features)

        # CNN 1D sul residuo antigene (tempo)
        x = self.conv1(ag_flat)  # (B, X, 64)
        x = self.gru_ag(x)  # (B, hidden_dim*2)
        x = self.dense_emb(x)  # (B, hidden_dim)

        # Positional embeddings per anticorpo residui da generare
        pos = tf.range(self.max_residues)
        pos_embed = self.pos_embedding(pos)  # (max_residues, hidden_dim)
        pos_embed = tf.expand_dims(pos_embed, 0)
        pos_embed = tf.tile(pos_embed, [B, 1, 1])  # (B, max_residues, hidden_dim)

        # Espandi embedding antigene e somma con pos embed
        x_exp = tf.expand_dims(x, 1)  # (B,1,hidden_dim)
        dec_input = x_exp + pos_embed  # (B, max_residues, hidden_dim)

        # Decoder RNN
        h = self.gru_dec(dec_input)  # (B, max_residues, hidden_dim)
        out = self.dense_out(h)  # (B, max_residues, 5*34)
        out = tf.reshape(out, [B, self.max_residues, self.atom_num, self.feature_dim])  # (B, max_res, 5, 34)

        # Mask out residui oltre K
        mask = tf.sequence_mask(K, maxlen=self.max_residues, dtype=tf.float32)
        mask = tf.reshape(mask, [B, self.max_residues, 1, 1])
        out = out * mask

        return out, K
