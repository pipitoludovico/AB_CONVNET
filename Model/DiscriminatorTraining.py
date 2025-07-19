from keras.models import load_model
from keras.optimizers import Adam
from Model.Models import Discriminator
from Model.CallBacks import lr_reduction, early_stopping, checkpoint
import joblib

import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt


def Train(args):
    if os.path.exists("best_model.keras"):
        print("Found existing model. Loading...")
        model = load_model("best_model.keras", compile=False, safe_mode=False)

        # Load scalers
        feature_scaler = joblib.load('feature_scaler.pkl')
        label_scaler = joblib.load('label_scaler.pkl')

        print("Loading and scaling data...")
        data = np.load("./matrices/padded_dataset.npz", allow_pickle=True)
        ab = data['ab']
        ag = data['ag']

        gbsa = data['gbsa'].reshape(-1, 1)

        continuous_idx = slice(0, 3)

        ab_cont = ab[..., continuous_idx].reshape(-1, 3)
        ag_cont = ag[..., continuous_idx].reshape(-1, 3)

        ab_scaled = feature_scaler.transform(ab_cont).reshape(ab.shape[0], ab.shape[1], ab.shape[2], 3)
        ag_scaled = feature_scaler.transform(ag_cont).reshape(ag.shape[0], ag.shape[1], ag.shape[2], 3)

        ab[..., continuous_idx] = ab_scaled
        ag[..., continuous_idx] = ag_scaled

        # ab = np.expand_dims(ab, axis=3)  # from (batch, 92, 5, 34) -> (batch, 92, 5, 1, 34)
        # ag = np.expand_dims(ag, axis=3)  # same for ag

        print(f"ab shape: {ab.shape}")
        print(f"ag shape: {ag.shape}")

        gbsa_scaled = label_scaler.transform(gbsa)

        dataset = tf.data.Dataset.from_tensor_slices((
            {'ab_input': ab, 'ag_input': ag},
            gbsa_scaled
        )).batch(args['batch']).prefetch(tf.data.AUTOTUNE)

        print("Evaluating...")
        gbsa_pred_scaled, validity_pred = model.predict(dataset)

        # If you're only inverse transforming the GBSA output
        gbsa_pred = label_scaler.inverse_transform(gbsa_pred_scaled)

        gbsa_original = label_scaler.inverse_transform(gbsa_scaled)
        mae = mean_absolute_error(gbsa_original, gbsa_pred)
        print(f"Mean Absolute Error (MAE): {mae:.4f}")

        print("Saving plots...")
        plt.figure(figsize=(6, 6))
        plt.scatter(gbsa_original, gbsa_pred, alpha=0.5)
        plt.xlabel("True GBSA")
        plt.ylabel("Predicted GBSA")
        plt.title("Predicted vs True GBSA")
        plt.plot([gbsa_original.min(), gbsa_original.max()],
                 [gbsa_original.min(), gbsa_original.max()], 'r--')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("predicted_vs_true_gbsa.png", dpi=300)
        plt.close()
        residuals = gbsa_original - gbsa_pred

        plt.figure(figsize=(6, 4))
        plt.hist(residuals, bins=50, color='skyblue', edgecolor='black')
        plt.title("Residuals (True - Predicted)")
        plt.xlabel("Error")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("residuals_histogram.png", dpi=300)
        plt.close()

    else:
        print("No existing model found. Training from scratch...")

        print("Data load")
        data = np.load("./matrices/padded_dataset.npz", allow_pickle=True)
        ab = data['ab']
        ag = data['ag']
        gbsa = data['gbsa'].reshape(-1, 1)

        print("Scaling x, y, z only")
        continuous_idx = slice(0, 3)
        ab_cont = ab[..., continuous_idx].reshape(-1, 3)
        ag_cont = ag[..., continuous_idx].reshape(-1, 3)

        feature_scaler = StandardScaler()
        feature_scaler.fit(np.vstack([ab_cont, ag_cont]))

        ab[..., continuous_idx] = feature_scaler.transform(ab_cont).reshape(ab.shape[0], ab.shape[1], ab.shape[2], 3)
        ag[..., continuous_idx] = feature_scaler.transform(ag_cont).reshape(ag.shape[0], ag.shape[1], ag.shape[2], 3)

        label_scaler = StandardScaler()
        gbsa_scaled = label_scaler.fit_transform(gbsa)

        print("Calling the dataset")
        # Create dummy labels for the diversity calculator output (won't be used for training)
        diversity_labels = np.zeros_like(gbsa_scaled)

        # Ensure data types are float32
        ab = ab.astype(np.float32)
        ag = ag.astype(np.float32)
        gbsa_scaled = gbsa_scaled.astype(np.float32)
        diversity_labels = diversity_labels.astype(np.float32)

        model = Discriminator()

        # Create dataset with the correct output names
        dataset = tf.data.Dataset.from_tensor_slices((
            {'ab_input': ab, 'ag_input': ag},
            {'gbsa_prediction': gbsa_scaled, 'diversity_calculator': diversity_labels}
        ))
        val_size = int(0.2 * len(gbsa_scaled))
        val_dataset = dataset.take(val_size).batch(args['batch']).prefetch(tf.data.AUTOTUNE)
        train_dataset = dataset.skip(val_size).batch(args['batch']).prefetch(tf.data.AUTOTUNE)

        print("Compiling the model...")
        optimizer = Adam(learning_rate=args["lr"])

        # OPTION 1: Set loss weight to 0 for diversity_calculator
        model.compile(
            optimizer=optimizer,
            loss={
                'gbsa_prediction': 'mse',
                'diversity_calculator': 'mse'
            },
            loss_weights={
                'gbsa_prediction': 1.0,
                'diversity_calculator': 0.0  # Don't train on this output
            },
            metrics={
                'gbsa_prediction': ['mae'],
                'diversity_calculator': ['mae']
            }
        )
        print("Training begins")
        model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=args['epoch'],
            callbacks=[checkpoint, early_stopping, lr_reduction],
            verbose=1,
        )

        print("Saving scalers")
        joblib.dump(feature_scaler, 'feature_scaler.pkl')
        joblib.dump(label_scaler, 'label_scaler.pkl')
