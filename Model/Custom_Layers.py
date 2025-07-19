from keras import layers
import tensorflow as tf


class Separator(layers.Layer):
    def __init__(self, **kwargs):
        super(Separator, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs[:, :, :, :8]  # Restituisce shape: (batch, 92, 5, 8)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(Separator, self).get_config()
        config.update()
        return config


class DiversityCalculator(layers.Layer):
    """Calcola la diversità (Shannon entropy normalizzata) per ogni sample del batch."""

    def __init__(self, **kwargs):
        super(DiversityCalculator, self).__init__(**kwargs)

    def call(self, inputs):
        aa_onehot = inputs[:, :, 0, -22:]  # shape: (batch, residues, 22)
        aa_counts = tf.reduce_sum(aa_onehot, axis=1)  # shape: (batch, 22)
        total_residues = tf.reduce_sum(aa_counts, axis=1, keepdims=True)  # shape: (batch, 1)
        aa_frequencies = aa_counts / (total_residues + 1e-8)  # shape: (batch, 22)
        entropy = -tf.reduce_sum(aa_frequencies * tf.math.log(aa_frequencies + 1e-8), axis=1)
        max_entropy = tf.math.log(tf.constant(22.0, dtype=entropy.dtype))
        normalized_entropy = entropy / max_entropy  # shape: (batch,)
        return tf.expand_dims(normalized_entropy, axis=1)  # shape: (batch, 1)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(DiversityCalculator, self).get_config()
        config.update()
        return config
