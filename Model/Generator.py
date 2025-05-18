import tensorflow as tf
from keras import layers, Model
from keras.layers import MultiHeadAttention, LayerNormalization


# Replace the shared Sequential block with a custom subclassed model
class SharedBlock(layers.Layer):
    def __init__(self):
        super().__init__()
        self.dense1 = layers.Dense(1024)
        self.reshape = layers.Reshape((16, 64))  # New shape for attention
        self.attn = MultiHeadAttention(num_heads=4, key_dim=16)
        self.norm = LayerNormalization()
        self.dense2 = layers.Dense(4096)
        self.leaky = layers.LeakyReLU(0.2)

    def call(self, x):
        x = self.dense1(x)
        x = self.reshape(x)  # Shape: (batch, 16, 64)
        attn_output = self.attn(x, x)  # Self-attention
        x = self.norm(x + attn_output)  # Residual connection
        x = layers.Flatten()(x)
        x = self.dense2(x)
        return self.leaky(x)


class ConditionalGenerator(Model):
    def __init__(self,
                 noise_dim=256,
                 label_dim=1,
                 ag_len=97,
                 atom_n=5,
                 coord_dim=3,
                 eleTypes=None,  # list of element type strings
                 amino_acids=None,  # list of residue names
                 temperature=0.8
                 ):
        super().__init__()
        assert eleTypes and amino_acids

        self.noise_dim = noise_dim
        self.label_dim = label_dim
        self.ag_len = ag_len
        self.atom_n = atom_n
        self.coord_dim = coord_dim
        self.temperature = temperature

        self.N_ele = len(eleTypes)
        self.N_aa = len(amino_acids)

        # Output dimensions
        self.flat_coords = ag_len * atom_n * coord_dim
        self.flat_ele = ag_len * atom_n * self.N_ele
        self.flat_aa = ag_len * atom_n * self.N_aa
        self.flat_charge = ag_len * atom_n * 1

        self.label_embedding = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(128)
        ])

        self.shared = SharedBlock()

        self.coord_head = tf.keras.Sequential([
            layers.Dense(512), layers.LeakyReLU(0.2),
            layers.Dense(self.flat_coords, activation='linear')  # normalized to [-1, 1]
        ])

        self.charge_head = tf.keras.Sequential([
            layers.Dense(512), layers.LeakyReLU(0.2),
            layers.Dense(self.flat_charge, activation='tanh')
        ])

        self.ele_head = tf.keras.Sequential([
            layers.Dense(512), layers.LeakyReLU(0.2),
            layers.Dense(self.flat_ele)  # logits → softmax later
        ])

        self.aa_head = tf.keras.Sequential([
            layers.Dense(512), layers.LeakyReLU(0.2),
            layers.Dense(self.flat_aa)  # logits → softmax later
        ])

    def call(self, noise, labels):
        label_embed = self.label_embedding(labels)
        x = tf.concat([noise, label_embed], axis=1)
        x = self.shared(x)
        return self.build_outputs(x)

    def build_outputs(self, x):
        batch_size = tf.shape(x)[0]

        # Coordinates
        c = self.coord_head(x)
        c = tf.reshape(c, (batch_size, self.ag_len, self.atom_n, self.coord_dim))

        # Partial charge
        q = self.charge_head(x)
        q = tf.reshape(q, (batch_size, self.ag_len, self.atom_n, 1))

        # Element types
        e_logits = self.ele_head(x)
        e_logits = tf.reshape(e_logits, (batch_size, self.ag_len, self.atom_n, self.N_ele))
        e = tf.nn.softmax(e_logits / self.temperature)

        # Amino acids
        a_logits = self.aa_head(x)
        a_logits = tf.reshape(a_logits, (batch_size, self.ag_len, self.atom_n, self.N_aa))
        a = tf.nn.softmax(a_logits / self.temperature)

        return tf.concat([c, q, e, a], axis=-1)
