import tensorflow as tf
from keras import layers


class Split(layers.Layer):
    def __init__(self, start, end, **kwargs):
        super().__init__(**kwargs)
        self.start = start
        self.end = end

    def call(self, inputs):
        return inputs[..., self.start:self.end]

    def get_config(self):
        config = super().get_config()
        config.update({"start": self.start,
                       "end": self.end})
        return config

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        if output_shape and len(output_shape) > 0:
            output_shape[-1] = self.end - self.start
        return tuple(output_shape)


class OneHotArgmax(layers.Layer):
    def __init__(self, depth=22, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.depth = depth
        self.axis = axis

    def call(self, inputs):
        residue_logits = tf.reduce_mean(inputs, axis=2)  # (batch, 92, 22)
        argmax_indices = tf.argmax(residue_logits, axis=-1)  # (batch, 92)
        one_hot_residue = tf.one_hot(argmax_indices, depth=self.depth, dtype=tf.float32)  # (batch, 92, 22)
        one_hot_expanded = tf.expand_dims(one_hot_residue, axis=2)  # (batch, 92, 1, 22)
        one_hot_atoms = tf.tile(one_hot_expanded, [1, 1, 5, 1])  # (batch, 92, 5, 22)
        return one_hot_atoms

    def get_config(self):
        config = super().get_config()
        config.update({'depth': self.depth, 'axis': self.axis})
        return config


class GumbelSoftmaxResidue(layers.Layer):
    def __init__(self, temperature=1.0, hard=False, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.hard = hard

    def call(self, inputs, training=None):
        # Input shape: (batch, 92, 5, 22)
        residue_logits = tf.reduce_sum(inputs, axis=2)  # (batch, 92, 22)

        if training:
            # Durante training, usa Gumbel Softmax per mantenere la differenziabilità
            gumbel_softmax = self._gumbel_softmax(residue_logits, self.temperature, self.hard)
        else:
            argmax_indices = tf.argmax(residue_logits, axis=-1)
            gumbel_softmax = tf.one_hot(argmax_indices, depth=22, dtype=tf.float32)

        one_hot_expanded = tf.expand_dims(gumbel_softmax, axis=2)  # (batch, 92, 1, 22)
        one_hot_atoms = tf.tile(one_hot_expanded, [1, 1, 5, 1])  # (batch, 92, 5, 22)

        return one_hot_atoms

    def _gumbel_softmax(self, logits, temperature, hard):
        # Gumbel noise
        gumbel_noise = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(logits), 0, 1)))

        # Gumbel softmax
        y = tf.nn.softmax((logits + gumbel_noise) / temperature)

        if hard:
            # Straight-through estimator
            y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, axis=-1, keepdims=True)), y.dtype)
            y = tf.stop_gradient(y_hard - y) + y

        return y

    def get_config(self):
        config = super().get_config()
        config.update({
            'temperature': self.temperature,
            'hard': self.hard
        })
        return config
