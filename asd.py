import tensorflow as tf
from tensorflow.keras import layers, Model


def Generator(input_shape):
    inputs = layers.Input(shape=input_shape)  # e.g., (X, 5, 34)

    # Example: flatten last dims and do some Dense layers
    x = layers.Flatten()(inputs)  # (batch, X*5*34)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)

    # Output continuous values for each atom: 5 atoms * 4 floats (x,y,z,q) per atom
    output_dim = input_shape[1] * input_shape[2] * 4  # X * 5 * 4
    continuous_output = layers.Dense(output_dim)(x)  # no activation or linear

    # Reshape to (batch, X, 5, 4)
    continuous_output = layers.Reshape((input_shape[1], input_shape[2], 4))(continuous_output)

    return Model(inputs=inputs, outputs=continuous_output)


# Usage:
input_shape = (10, 5, 34)  # example
model = Generator(input_shape)

dummy_input = tf.random.normal((1,) + input_shape)
continuous_out = model(dummy_input)
print("Output shape:", continuous_out.shape)  # (1, 10, 5, 4)
print(continuous_out)