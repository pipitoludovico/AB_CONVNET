import tensorflow as tf
from keras import layers, Model


class ConditionalGenerator(Model):
    def __init__(self,
                 noise_dim=100,
                 label_dim=1,
                 ag_len=97,
                 atom_n=5,
                 coord_dim=3,
                 eleTypes=None,  # list of 8 strings
                 amino_acids=None  # list of 22 strings
                 ):
        super().__init__()
        assert eleTypes and amino_acids
        self.noise_dim = noise_dim
        self.input_dim = noise_dim + label_dim

        self.ag_len = ag_len
        self.atom_n = atom_n
        self.coord_dim = coord_dim

        self.N_ele = len(eleTypes)
        self.N_aa = len(amino_acids)

        # Shared trunk
        self.shared1 = layers.Dense(512, activation='relu')
        self.shared2 = layers.Dense(1024, activation='relu')

        flat_coords = ag_len * atom_n * coord_dim
        flat_ele = ag_len * atom_n * self.N_ele
        flat_aa = ag_len * atom_n * self.N_aa
        flat_charge = ag_len * atom_n * 1

        # Heads
        self.coord_head = layers.Dense(flat_coords, activation='tanh', name='coords')
        self.ele_head = layers.Dense(flat_ele, activation=None, name='ele_logits')
        self.aa_head = layers.Dense(flat_aa, activation=None, name='aa_logits')
        self.charge_head = layers.Dense(flat_charge, activation='tanh', name='charge')  # learnable

    def call(self, noise, labels):
        x = tf.concat([noise, labels], axis=1)
        x = self.shared1(x)
        x = self.shared2(x)

        c = self.coord_head(x)
        c = tf.reshape(c, (-1, self.ag_len, self.atom_n, self.coord_dim))

        e_logits = self.ele_head(x)
        e_logits = tf.reshape(e_logits, (-1, self.ag_len, self.atom_n, self.N_ele))
        e = tf.argmax(e_logits, axis=-1)
        e = tf.stop_gradient(tf.one_hot(e, depth=self.N_ele, dtype=tf.float32))

        a_logits = self.aa_head(x)
        a_logits = tf.reshape(a_logits, (-1, self.ag_len, self.atom_n, self.N_aa))
        a = tf.argmax(a_logits, axis=-1)
        a = tf.stop_gradient(tf.one_hot(a, depth=self.N_aa, dtype=tf.float32))

        q = self.charge_head(x)
        q = tf.reshape(q, (-1, self.ag_len, self.atom_n, 1))  # shape: (..., 1)

        out = tf.concat([c, q, e, a], axis=-1)
        return out
