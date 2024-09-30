import numpy as np
from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers import Conv1D, GaussianDropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sys import maxsize

np.set_printoptions(threshold=maxsize)


def BuildLinearModel(X_train, ConvNet=False, l2_rate: float = 0.01):
    input_shape = X_train.shape[1:]
    inputs = Input(shape=input_shape)
    if not ConvNet:
        x = Flatten()(inputs)
        x = Dense(64, activation='sigmoid')(x)
        x = Dense(128, activation='sigmoid')(x)
        x = Dense(128, activation='sigmoid', bias_regularizer=l2(l2_rate))(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1)(x)

        model_ = Model(inputs=inputs, outputs=outputs)
        return model_
    else:
        x = Conv1D(64, kernel_size=6, activation='relu')(inputs)
        x = Conv1D(128, kernel_size=6, activation='relu')(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu', bias_regularizer=l2(l2_rate))(x)
        x = GaussianDropout(0.2)(x)
        outputs = Dense(1)(x)

        model_ = Model(inputs=inputs, outputs=outputs)
        return model_


def TrainModel(args_):
    dataset = np.load('matrices/padded.npy', allow_pickle=True)
    max_for_scaling = np.max(dataset)
    split_percent = float(args_["split"] / 100)
    batch_size = args_["batch"]
    epochs = args_["epoch"]
    conv = args_["conv"]
    l2_rate = args_["l2"]
    training = dataset[:int(len(dataset) * split_percent)]
    test = dataset[int(len(dataset) * split_percent):]
    X_train = training[:, :, :-1]
    y_train = training[:, :, -1] / max_for_scaling

    X_test = test[:, :, :-1]
    y_test = test[:, :, -1] / max_for_scaling
    model = BuildLinearModel(X_train, ConvNet=conv, l2_rate=l2_rate)
    model.compile(optimizer=Adam(learning_rate=args_["lr"]), loss='mean_squared_error')
    checkpoint = ModelCheckpoint(f'{args_["name"]}_lr{args_["lr"]}_l2{args_["l2"]}_best.keras', monitor='val_loss',
                                 save_best_only=True, mode='min', verbose=1)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test),
              callbacks=[checkpoint])
