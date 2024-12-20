import os.path
import numpy as np
from os import listdir, walk
from sys import maxsize
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback

np.set_printoptions(threshold=maxsize)


def BuildDenseNet(X_train, l2_rate: float = 0.01):
    input_shape = X_train.shape[1:]  # (5200,)
    inputs = Input(shape=input_shape)

    # Add fully connected layers (Dense layers)
    x = Dense(256, activation='relu', kernel_regularizer=l2(l2_rate))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)  # Dropout for regularization

    x = Dense(128, activation='relu', kernel_regularizer=l2(l2_rate))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(64, activation='relu', kernel_regularizer=l2(l2_rate))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # Final output layer
    outputs = Dense(1)(x)

    model_ = Model(inputs=inputs, outputs=outputs)
    return model_


def TrainModel(args_):
    dataset = np.load('matrices/padded.npy', allow_pickle=True)
    max_for_scaling = np.min(dataset[:, -1])  # We use min because the GBSA is negative!!!
    print("ORIGINAL DATASET", dataset.shape, "MEAN", np.mean(dataset), "STD", np.std(dataset), "MIN", np.min(dataset),
          "MAX", np.max(dataset), "SCALING LABEL MIN", max_for_scaling)
    # max_for_scaling = np.max(np.abs(dataset))
    split_percent = 1 - float(args_["split"] / 100)
    batch_size = args_["batch"]
    epochs = args_["epoch"]
    l2_rate = args_["l2"]
    X = dataset[:, :-1]
    y = dataset[:, -1] / max_for_scaling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_percent, random_state=42)
    print('X_train', X_train.shape, "X_test", X_test.shape, "y_train", y_train.shape, "y_test", y_test.shape, "max",
          max_for_scaling)

    # Build the Dense Model
    model = BuildDenseNet(X_train, l2_rate=l2_rate)
    model.compile(optimizer=Adam(learning_rate=args_["lr"]), loss="mean_absolute_error")

    # Checkpoint and EarlyStopping callbacks
    checkpoint = ModelCheckpoint(f'{args_["name"]}_lr{args_["lr"]}_l2{args_["l2"]}_best.keras', monitor='val_loss',
                                 save_best_only=True, mode='min', verbose=0)

    early_stop = EarlyStopping(monitor='val_loss', patience=20)

    class prediction_history_DENSE(Callback):
        def on_epoch_end(self, epoch, logs=None):
            differences = []  # List to store differences for each sample
            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1} - Predictions and Differences:")

                for i in range(len(X_train)):
                    # Prepare the sample for prediction
                    sample = np.expand_dims(X_train[i], axis=0)  # Add batch dimension
                    prediction = model.predict(sample)

                    # Rescale prediction and target for comparison
                    predicted_rescaled = prediction * max_for_scaling
                    actual_rescaled = y_train[i] * max_for_scaling

                    # Compute the difference
                    difference = predicted_rescaled - actual_rescaled
                    differences.append(difference)

                    # Print details for this sample
                    print(
                        f"Sample {i + 1}: Predicted = {predicted_rescaled[0][0]:.4f} Actual = {actual_rescaled:.4f}, Difference = {difference[0][0]:.4f}\n")

                # Print the average difference across all samples for the epoch
                avg_difference = np.mean(differences)
                print(f"\nAverage Difference for Epoch {epoch + 1}: {avg_difference:.4f}\n")

    # Start training with the model
    ph = prediction_history()
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test),
                        callbacks=[checkpoint, ph, early_stop], verbose='auto')

    # Plot the loss curve
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{args_["name"]}_loss_plot.png')


def Test():
    loaded_model = None
    dataset = np.load('matrices/padded.npy', allow_pickle=True)
    scaling_factor = np.min(dataset[:, -1])
    print(np.min(dataset), np.max(dataset), np.std(dataset))
    real = None
    print("\n", scaling_factor, "as a scaling factor.\n")

    for model in listdir('.'):
        if model.endswith(".keras"):
            loaded_model = load_model(model)

    for folder, _, sample in walk('./predictions'):
        for file in sample:
            if file.endswith('npy'):
                name = file.split(".")[0]
                sample = np.load(os.path.join(folder, file))
                original_shape = sample.shape
                sample = np.expand_dims(sample, axis=0)
                print(folder, file, sample.shape)
                if sample.shape[1] == 5200:
                    prediction = loaded_model.predict(sample)
                    prediction_rescaled = prediction * scaling_factor
                    with open('comparazione', 'r') as comparazione:
                        for line in comparazione.readlines():
                            if name.split('.')[0] in line:
                                real = line.split()[3]
                    print(name, original_shape, prediction, prediction_rescaled, real)
