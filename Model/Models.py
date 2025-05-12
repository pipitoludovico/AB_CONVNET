from keras import layers, Model, Input


def Net(ab_shape=(92, 5, 34), ag_shape=(97, 5, 34)):
    ab_input = Input(shape=ab_shape, name="ab_input")  # (residues, atoms, features)
    ag_input = Input(shape=ag_shape, name="ag_input")
    gbsa_input = Input(shape=(1,), name="gbsa_input")

    def encode_seq(seq_input):
        # Add a channel dimension so shape becomes (residues, atoms, features, 1)
        x = layers.Reshape((seq_input.shape[1], seq_input.shape[2], seq_input.shape[3], 1))(seq_input)
        x = layers.TimeDistributed(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))(x)
        x = layers.TimeDistributed(layers.Flatten())(x)
        x = layers.GRU(128)(x)
        return x

    x_ab = encode_seq(ab_input)
    x_ag = encode_seq(ag_input)

    x = layers.Concatenate()([x_ab, x_ag, gbsa_input])
    x = layers.Dense(128, activation='relu')(x)

    # Two heads: real/fake classification and GBSA regression
    validity = layers.Dense(1, activation='sigmoid', name='validity')(x)
    gbsa_pred = layers.Dense(1, activation='linear', name='gbsa_pred')(x)

    return Model(inputs=[ab_input, ag_input, gbsa_input], outputs=[validity, gbsa_pred], name="Discriminator")
