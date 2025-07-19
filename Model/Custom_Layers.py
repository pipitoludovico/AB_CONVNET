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


class ABAgInteractionLayer(layers.Layer):
    """Models direct AB-AG interactions through attention mechanism."""

    def __init__(self, hidden_dim=64, **kwargs):
        super(ABAgInteractionLayer, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        # input_shape: [(batch, ab_len, atoms, features), (batch, ag_len, atoms, features)]
        super(ABAgInteractionLayer, self).build(input_shape)
        ab_shape, ag_shape = input_shape
        feature_dim = ab_shape[-1]

        # Create and build projection layers for attention
        self.ab_query = layers.Dense(self.hidden_dim, name='ab_query')
        self.ag_key = layers.Dense(self.hidden_dim, name='ag_key')
        self.ag_value = layers.Dense(self.hidden_dim, name='ag_value')
        self.output_proj = layers.Dense(feature_dim, name='output_proj')

        # Explicitly build the Dense layers
        flattened_shape = (ab_shape[0], ab_shape[1] * 5, ab_shape[3])
        self.ab_query.build(flattened_shape)
        self.ag_key.build(flattened_shape)
        self.ag_value.build(flattened_shape)
        self.output_proj.build((flattened_shape[0], flattened_shape[1], self.hidden_dim))

    def call(self, inputs):
        ab_input, ag_input = inputs
        batch_size = tf.shape(ab_input)[0]
        ab_len = tf.shape(ab_input)[1]
        ag_len = tf.shape(ag_input)[1]

        # Flatten spatial dimensions for attention computation
        ab_flat = tf.reshape(ab_input, [batch_size, ab_len * 5, -1])  # (batch, ab_len*atoms, features)
        ag_flat = tf.reshape(ag_input, [batch_size, ag_len * 5, -1])  # (batch, ag_len*atoms, features)

        # Create masks for valid positions
        ab_mask = tf.reduce_sum(tf.abs(ab_flat), axis=-1) > 0  # (batch, ab_len*atoms)
        ag_mask = tf.reduce_sum(tf.abs(ag_flat), axis=-1) > 0  # (batch, ag_len*atoms)

        # Compute attention: AB queries attend to AG keys/values
        queries = self.ab_query(ab_flat)  # (batch, ab_len*atoms, hidden_dim)
        keys = self.ag_key(ag_flat)  # (batch, ag_len*atoms, hidden_dim)
        values = self.ag_value(ag_flat)  # (batch, ag_len*atoms, hidden_dim)

        # Attention scores
        attention_scores = tf.matmul(queries, keys, transpose_b=True)  # (batch, ab_len*atoms, ag_len*atoms)
        attention_scores = attention_scores / tf.sqrt(tf.cast(self.hidden_dim, tf.float32))

        # Apply AG mask to attention scores
        ag_mask_expanded = tf.expand_dims(ag_mask, axis=1)  # (batch, 1, ag_len*atoms)
        attention_scores = tf.where(ag_mask_expanded, attention_scores, -1e9)

        # Softmax attention weights
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)  # (batch, ab_len*atoms, ag_len*atoms)

        # Apply AB mask to attention weights
        ab_mask_expanded = tf.expand_dims(ab_mask, axis=-1)  # (batch, ab_len*atoms, 1)
        attention_weights = tf.where(ab_mask_expanded, attention_weights, 0.0)

        # Weighted sum of AG values
        interaction_features = tf.matmul(attention_weights, values)  # (batch, ab_len*atoms, hidden_dim)

        # Project back to original feature dimension
        interaction_features = self.output_proj(interaction_features)  # (batch, ab_len*atoms, features)

        # Reshape back to original AB shape
        interaction_features = tf.reshape(interaction_features, [batch_size, ab_len, 5, -1])

        # Add residual connection with original AB input
        return ab_input + interaction_features

    def compute_output_shape(self, input_shape):
        return input_shape[0]  # Same shape as AB input

    def get_config(self):
        config = super(ABAgInteractionLayer, self).get_config()
        config.update({'hidden_dim': self.hidden_dim})
        return config


class PositionalAttentionPooling(layers.Layer):
    """Attention-based pooling that considers residue positions."""

    def __init__(self, hidden_dim=128, **kwargs):
        super(PositionalAttentionPooling, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        super(PositionalAttentionPooling, self).build(input_shape)
        # input_shape: (batch, residues, atoms, features)

        # Create the Dense layers
        self.attention_proj = layers.Dense(self.hidden_dim, activation='tanh')
        self.attention_weights = layers.Dense(1, activation=None)

        # Explicitly build the Dense layers
        flattened_shape = (input_shape[0], input_shape[1] * input_shape[2], input_shape[3])
        self.attention_proj.build(flattened_shape)
        self.attention_weights.build((flattened_shape[0], flattened_shape[1], self.hidden_dim))

    def call(self, inputs):
        # inputs shape: (batch, residues, atoms, features)
        batch_size = tf.shape(inputs)[0]
        n_residues = tf.shape(inputs)[1]
        n_atoms = tf.shape(inputs)[2]

        # Reshape for processing
        inputs_flat = tf.reshape(inputs, [batch_size, n_residues * n_atoms, -1])

        # Create position mask
        mask = tf.reduce_sum(tf.abs(inputs_flat), axis=-1) > 0  # (batch, residues*atoms)

        # Compute attention scores
        attention_input = self.attention_proj(inputs_flat)  # (batch, residues*atoms, hidden_dim)
        attention_scores = self.attention_weights(attention_input)  # (batch, residues*atoms, 1)
        attention_scores = tf.squeeze(attention_scores, axis=-1)  # (batch, residues*atoms)

        # Apply mask to attention scores
        attention_scores = tf.where(mask, attention_scores, -1e9)

        # Softmax to get attention weights
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)  # (batch, residues*atoms)
        attention_weights = tf.expand_dims(attention_weights, axis=-1)  # (batch, residues*atoms, 1)

        # Weighted sum
        pooled = tf.reduce_sum(inputs_flat * attention_weights, axis=1)  # (batch, features)

        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    def get_config(self):
        config = super(PositionalAttentionPooling, self).get_config()
        config.update({'hidden_dim': self.hidden_dim})
        return config


class ResiduePositionMask(layers.Layer):
    """Creates position mask for AB sequence at residue level."""

    def call(self, inputs):
        # inputs shape: (batch, residues, atoms, features)
        # Create mask based on non-zero values at residue level
        residue_sums = tf.reduce_sum(tf.abs(inputs), axis=[2, 3])  # (batch, residues)
        mask = tf.cast(residue_sums > 0, tf.float32)  # (batch, residues)
        return tf.expand_dims(tf.expand_dims(mask, axis=2), axis=3)  # (batch, residues, 1, 1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], 1, 1
