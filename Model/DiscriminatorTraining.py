from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from Model.Models import Discriminator
from Model.CallBacks import lr_reduction, early_stopping
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
        validity_labels = np.ones_like(gbsa_scaled)

        dataset = tf.data.Dataset.from_tensor_slices((
            {'ab_input': ab, 'ag_input': ag},
            {'gbsa_prediction': gbsa_scaled, 'validity': validity_labels}
        ))
        val_size = int(0.1 * len(gbsa_scaled))
        val_dataset = dataset.take(val_size).batch(args['batch']).prefetch(tf.data.AUTOTUNE)
        train_dataset = dataset.skip(val_size).batch(args['batch']).prefetch(tf.data.AUTOTUNE)

        print("Compiling the model...")
        model = Discriminator(ab_shape=ab.shape[1:], ag_shape=ag.shape[1:])
        optimizer = Adam(learning_rate=args["lr"])
        model.compile(
            optimizer=optimizer,
            loss={
                'gbsa_prediction': 'mse',
                'validity': 'binary_crossentropy'
            },
            metrics={
                'gbsa_prediction': ['mae'],
                'validity': ['accuracy']
            }
        )

        best_ckpt = ModelCheckpoint(
            filepath='best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )

        y_real = {
            "gbsa_prediction": gbsa_scaled,
            "validity": np.ones_like(gbsa_scaled),
        }
        sample_weights_real = {
            "gbsa_prediction": np.ones_like(gbsa_scaled).flatten(),  # count toward loss
            "validity": np.ones_like(gbsa_scaled).flatten(),
        }

        y_fake = {
            "gbsa_prediction": np.zeros_like(gbsa_scaled),  # dummy
            "validity": np.zeros_like(gbsa_scaled),  # label as fake
        }
        sample_weights_fake = {
            "gbsa_prediction": np.zeros_like(gbsa_scaled).flatten(),  # ignore loss
            "validity": np.ones_like(gbsa_scaled).flatten(),
        }

        print("Training begins")
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=args['epoch'],
            callbacks=[best_ckpt, early_stopping, lr_reduction],
            verbose=1,
        )

        print("Saving scalers")
        joblib.dump(feature_scaler, 'feature_scaler.pkl')
        joblib.dump(label_scaler, 'label_scaler.pkl')
