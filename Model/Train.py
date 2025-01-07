import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.losses import MeanAbsoluteError
from Model.Models import Net
from Model.DataPrep import PrepareData
from Model.CallBacks import lr_reduction
import joblib


def Train(args):
    """
    Train the model on the full dataset and study overfitting, focusing on Mean Absolute Error (MAE).

    Args:
    - args: Dictionary containing hyperparameters (l2, learning_rate, epoch, batch_size).

    Returns:
    - history: Training history object.
    """
    # Prepare the data
    # X_scaled, y_scaled, feature_scaler, label_scaler = PrepareData("./matrices/padded.npy")
    X_scaled, _, feature_scaler, _ = PrepareData("./matrices/padded.npy")
    # Create the model
    X = X_scaled[:, :-1]  # Prendi tutte le colonne tranne l'ultima (4800 feature)
    y = X_scaled[:, -1]  # Prendi l'ultima colonna (label)
    X = X[:, :, np.newaxis]  # Aggiungi una dimensione per le feature

    model = Net()

    # Compile the model with Mean Absolute Error loss
    optimizer = Adam(learning_rate=args["lr"])
    model.compile(optimizer=optimizer, loss=MeanAbsoluteError(), metrics=[MeanAbsoluteError()])

    # Callbacks: Optionally track model performance during training
    checkpoint = ModelCheckpoint("model.keras", monitor='loss', save_best_only=True, mode='min', verbose=1)
    early_stopping = EarlyStopping(monitor='loss', patience=100, restore_best_weights=True, verbose=1)

    # Train the model on the entire dataset, measuring only MAE
    history = model.fit(X, y, epochs=args['epoch'], batch_size=args["batch"], shuffle=False,
                        callbacks=[checkpoint, early_stopping, lr_reduction], verbose=1)

    # Save the feature and label scalers for later use
    joblib.dump(feature_scaler, 'feature_scaler.pkl')  # Save feature scaler
    # joblib.dump(label_scaler, 'label_scaler.pkl')  # Save label scaler
    print("Scalers saved to 'feature_scaler.pkl' and 'label_scaler.pkl'")

    # Evaluate the model on the same dataset to prove overfitting
    train_preds_scaled = model.predict(X)

    # Inverse transform the predictions and ground truth labels to the original scale
    train_preds = feature_scaler.inverse_transform(train_preds_scaled)
    y_original = feature_scaler.inverse_transform(y)

    # Calculate the Mean Absolute Error (MAE) on the original scale
    mae = np.mean(np.abs(y_original - train_preds))
    print(f"Mean Absolute Error (MAE) on training data (original scale): {mae:.4f}")

    return history
