from keras import layers, Model, Input


def Net(ab_shape, ag_shape):
    ab_input = Input(shape=ab_shape, name="ab_input")
    ag_input = Input(shape=ag_shape, name="ag_input")

    def encode_entity(x):
        x = layers.TimeDistributed(layers.TimeDistributed(layers.Dense(64, activation='relu')))(x)
        x = layers.Reshape((x.shape[1], -1))(x)
        x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x)
        x = layers.GlobalAveragePooling1D()(x)
        return x

    x_ab = encode_entity(ab_input)
    x_ag = encode_entity(ag_input)
    x = layers.Concatenate()([x_ab, x_ag])
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(1)(x)

    return Model(inputs=[ab_input, ag_input], outputs=out)
