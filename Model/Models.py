from keras.layers import Input, Dense, Flatten, MaxPooling1D, Reshape, Conv1D
from keras.models import Model


def Net():
    inputs = Input(shape=(None, 1))  # None consente input con timesteps variabili
    x = Reshape((50, 60))(inputs)  # Reshape to (50 res pairs, 60 features)

    x = Conv1D(filters=128, kernel_size=3, strides=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2, padding='same')(x)

    x = Conv1D(filters=256, kernel_size=6, strides=6, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2, padding='same')(x)

    x = Conv1D(filters=256, kernel_size=12, strides=12, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2, padding='same')(x)

    x = Conv1D(filters=512, kernel_size=24, strides=24, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2, padding='same')(x)

    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(1, activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model
