import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.optimizers import Adam
from keras.losses import Huber
from keras.callbacks import ModelCheckpoint
from Model.Models import Net
from sklearn.preprocessing import StandardScaler
from Model.CallBacks import lr_reduction, early_stopping
import joblib
from sklearn.metrics import mean_absolute_error
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def Train(args):
    if os.path.exists("best_model.keras"):
        print("==> Found existing model. Loading...")
        model = load_model("best_model.keras", compile=False)

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

        ab[..., continuous_idx] = feature_scaler.transform(ab_cont).reshape(ab.shape[0], ab.shape[1], ab.shape[2], 3)
        ag[..., continuous_idx] = feature_scaler.transform(ag_cont).reshape(ag.shape[0], ag.shape[1], ag.shape[2], 3)
        gbsa_scaled = label_scaler.transform(gbsa)

        dataset = tf.data.Dataset.from_tensor_slices((
            {'ab_input': ab, 'ag_input': ag, 'gbsa_input': gbsa_scaled},
            {'validity': np.ones((len(gbsa_scaled), 1)), 'gbsa_pred': gbsa_scaled}
        )).batch(args['batch']).prefetch(tf.data.AUTOTUNE)

        print("Evaluating...")

        preds_dict = model.predict(dataset)

        if isinstance(preds_dict, dict):
            preds_scaled = preds_dict['gbsa_pred']
        # If list: assume it's [validity_pred, gbsa_pred]
        elif isinstance(preds_dict, list) or isinstance(preds_dict, tuple):
            preds_scaled = preds_dict[1]
        else:
            raise TypeError("Unexpected prediction output format")

        if preds_scaled.ndim > 2:
            preds_scaled = preds_scaled.reshape(-1, 1)

        preds = label_scaler.inverse_transform(preds_scaled)
        gbsa_original = label_scaler.inverse_transform(gbsa_scaled)

        mae = mean_absolute_error(gbsa_original, preds)
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        return None

    else:
        print("==> No existing model found. Training from scratch...")

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

        validity_labels = np.ones((len(gbsa_scaled), 1), dtype=np.float32)

        print("Calling the dataset")
        dataset = tf.data.Dataset.from_tensor_slices((
            {'ab_input': ab, 'ag_input': ag, 'gbsa_input': gbsa_scaled},
            {'validity': validity_labels, 'gbsa_pred': gbsa_scaled}
        ))
        # Simple split: 90% train, 10% val
        val_size = int(0.1 * len(gbsa_scaled))
        val_dataset = dataset.take(val_size).batch(args['batch']).prefetch(tf.data.AUTOTUNE)
        train_dataset = dataset.skip(val_size).batch(args['batch']).prefetch(tf.data.AUTOTUNE)

        print("Compiling the model...")
        model = Net(ab_shape=ab.shape[1:], ag_shape=ag.shape[1:])
        optimizer = Adam(learning_rate=args["lr"])
        model.compile(
            optimizer=optimizer,
            loss={'validity': 'binary_crossentropy', 'gbsa_pred': Huber()},
            loss_weights={'validity': 1.0, 'gbsa_pred': 1.0}
        )

        best_ckpt = ModelCheckpoint(
            filepath='best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )

        print("Train begins")
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

        return history
