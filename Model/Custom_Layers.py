import tensorflow as tf
from tensorflow.keras import layers


class CoordinateExtractor(layers.Layer):
    def __init__(self, **kwargs):
        super(CoordinateExtractor, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs[..., :3]

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (3,)


class AtomExtractor(layers.Layer):
    def __init__(self, **kwargs):
        super(AtomExtractor, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs[..., 3:8]

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (5,)


class CoordinateSum(layers.Layer):
    def __init__(self, **kwargs):
        super(CoordinateSum, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.reduce_sum(tf.abs(inputs), axis=[2, 3], keepdims=True)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 1, 1)


class PaddingMask(layers.Layer):
    def __init__(self, **kwargs):
        super(PaddingMask, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.cast(inputs > 0.0, tf.float32)

    def compute_output_shape(self, input_shape):
        return input_shape


class MaskSqueezer(layers.Layer):
    def __init__(self, **kwargs):
        super(MaskSqueezer, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.squeeze(inputs, axis=2)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 1)


class LogitsMasker(layers.Layer):
    def __init__(self, **kwargs):
        super(LogitsMasker, self).__init__(**kwargs)

    def call(self, inputs):
        logits, mask = inputs
        return logits + (1.0 - mask) * (-1e9)

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class ResidueExpander(layers.Layer):
    def __init__(self, **kwargs):
        super(ResidueExpander, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=2)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (1,) + input_shape[-1:]


class ResidueTiler(layers.Layer):
    def __init__(self, atoms_per_res, **kwargs):
        super(ResidueTiler, self).__init__(**kwargs)
        self.atoms_per_res = atoms_per_res

    def call(self, inputs):
        return tf.tile(inputs, [1, 1, self.atoms_per_res, 1])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.atoms_per_res, input_shape[3])

    def get_config(self):
        config = super(ResidueTiler, self).get_config()
        config.update({'atoms_per_res': self.atoms_per_res})
        return config


class MaskTiler(layers.Layer):
    def __init__(self, atoms_per_res, num_residue_types, **kwargs):
        super(MaskTiler, self).__init__(**kwargs)
        self.atoms_per_res = atoms_per_res
        self.num_residue_types = num_residue_types

    def call(self, inputs):
        return tf.tile(inputs, [1, 1, self.atoms_per_res, self.num_residue_types])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.atoms_per_res, self.num_residue_types)

    def get_config(self):
        config = super(MaskTiler, self).get_config()
        config.update({
            'atoms_per_res': self.atoms_per_res,
            'num_residue_types': self.num_residue_types
        })
        return config


class GumbelSoftmax(layers.Layer):
    def __init__(self, temperature=1.0, hard=False, **kwargs):
        super(GumbelSoftmax, self).__init__(**kwargs)
        self.temperature = temperature
        self.hard = hard

    def call(self, inputs):
        gumbel_noise = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(inputs), 1e-20, 1.0)))
        y_soft = tf.nn.softmax((inputs + gumbel_noise) / self.temperature)

        if self.hard:
            y_hard = tf.one_hot(tf.argmax(y_soft, axis=-1), depth=tf.shape(inputs)[-1])
            y_soft = tf.stop_gradient(y_hard - y_soft) + y_soft

        return y_soft

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(GumbelSoftmax, self).get_config()
        config.update({
            'temperature': self.temperature,
            'hard': self.hard
        })
        return config
