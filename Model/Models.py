from keras import layers, Model, Input


def Net(ab_shape=(None, 5, 34), ag_shape=(None, 5, 34)):
    ab_input = Input(shape=ab_shape, name="ab_input")  # (residues, atoms, features)
    ag_input = Input(shape=ag_shape, name="ag_input")
    gbsa_input = Input(shape=(1,), name="gbsa_input")

    def encode_entity(entity_input):
        # Flatten atom-level features
        x = layers.TimeDistributed(layers.Flatten())(entity_input)  # (R, D)
        x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x)  # contextualize across residues
        x = layers.GlobalAveragePooling1D()(x)  # (D,)
        return x

    x_ab = encode_entity(ab_input)
    x_ag = encode_entity(ag_input)

    x = layers.Concatenate()([x_ab, x_ag, gbsa_input])
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    # Outputs
    validity = layers.Dense(1, activation='sigmoid', name='validity')(x)
    gbsa_pred = layers.Dense(1, activation='linear', name='gbsa_pred')(x)

    return Model(inputs=[ab_input, ag_input, gbsa_input], outputs=[validity, gbsa_pred], name="Discriminator")
