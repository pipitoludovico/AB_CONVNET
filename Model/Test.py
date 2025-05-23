from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import numpy as np
import joblib
import json
from os import path
from keras.models import load_model
from keras.optimizers import Adam
from keras.losses import Huber


def Test(args):
    """
    Test the CNN discriminator model using input matrices from ./predictions/*/
    Each sample directory should contain: abMatrix*.npy and agMatrix*.npy.
    """

    CONTINUOUS_IDX = slice(0, 3)  # x, y, z columns

    # Load pad config
    try:
        with open("matrices/pad_config.json", "r") as f:
            config = json.load(f)
        MAX_AB_LEN = config["max_ab_len"]
        MAX_AG_LEN = config["max_ag_len"]
        print(f"Loaded padding lengths: AB = {MAX_AB_LEN}, AG = {MAX_AG_LEN}")
    except Exception as e:
        print(f"Error loading pad_config.json: {e}")
        return

    # Load scalers
    try:
        feature_scaler = joblib.load('feature_scaler.pkl')
        label_scaler = joblib.load('label_scaler.pkl')
        print("Feature and label scalers loaded successfully.")
    except Exception as e:
        print(f"Scaler loading error: {e}")
        return

    # Load the model
    try:
        model = load_model(args['model'], compile=False)
        model.compile(
            optimizer=Adam(learning_rate=args['lr']),
            loss={'validity': 'binary_crossentropy', 'gbsa_pred': Huber()},
            loss_weights={'validity': 1.0, 'gbsa_pred': 1.0}
        )
        print("Model loaded and compiled.")
    except Exception as e:
        print(f"Model loading error: {e}")
        return

    print("\nPredictions:")
    print("-" * 80)
    print(f"{'Sample':<25} {'Predicted GBSA':<20} {'Predicted Validity':<20}")
    print("-" * 80)

    base_dir = './predictions'

    for sample_name in os.listdir(base_dir):
        sample_dir = path.join(base_dir, sample_name)
        ab_path = path.join(sample_dir, f"abMatrix{sample_name}.npy")
        ag_path = path.join(sample_dir, f"agMatrix{sample_name}.npy")

        if not (path.exists(ab_path) and path.exists(ag_path)):
            continue

        try:
            ab = np.load(ab_path)  # (R1, 5, 34)
            ag = np.load(ag_path)  # (R2, 5, 34)

            def pad_matrix(mat, target_len):
                current_len = mat.shape[0]
                if current_len < target_len:
                    padding = np.zeros((target_len - current_len, mat.shape[1], mat.shape[2]))
                    return np.concatenate([mat, padding], axis=0)
                return mat[:target_len]

            ab = pad_matrix(ab, MAX_AB_LEN)
            ag = pad_matrix(ag, MAX_AG_LEN)

            # Flatten and scale x, y, z features
            ab_flat = ab.reshape(-1, ab.shape[-1])
            ag_flat = ag.reshape(-1, ag.shape[-1])

            ab_flat[:, CONTINUOUS_IDX] = feature_scaler.transform(ab_flat[:, CONTINUOUS_IDX])
            ag_flat[:, CONTINUOUS_IDX] = feature_scaler.transform(ag_flat[:, CONTINUOUS_IDX])

            ab_scaled = ab_flat.reshape(1, *ab.shape)
            ag_scaled = ag_flat.reshape(1, *ag.shape)

            # Predict
            preds = model.predict([ab_scaled, ag_scaled], verbose=0)

            # Expecting [gbsa_pred_scaled, validity] or vice versa
            if isinstance(preds, (list, tuple)) and len(preds) == 2:
                gbsa_pred_scaled, validity = preds
            else:
                raise ValueError("Unexpected model output format.")

            # Reshape if needed
            if gbsa_pred_scaled.ndim > 2:
                gbsa_pred_scaled = gbsa_pred_scaled.reshape(gbsa_pred_scaled.shape[0], -1)

            # Inverse transform
            gbsa_pred = label_scaler.inverse_transform(gbsa_pred_scaled)[0, 0]
            validity_score = float(validity.flatten()[0])  # Assuming single sample

            print(f"{sample_name:<25} {gbsa_pred:<20.4f} {validity_score:<20.4f}")

        except Exception as e:
            print(f"Error with sample '{sample_name}': {e}")


def Test2(args):
    """
    Predict on the training data used during training to check how well the model fits.
    """
    # Load trained model
    try:
        model = load_model(args["model"])
        print(f"✅ Loaded model from {args['model']}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

    # Load scalers
    feature_scaler = joblib.load("feature_scaler.pkl")
    label_scaler = joblib.load("label_scaler.pkl")

    # Load the same training dataset used during training
    data = np.load("./matrices/padded_dataset.npz", allow_pickle=True)
    ab = data['ab']  # shape: (N, ab_len, 5, 34)
    ag = data['ag']  # shape: (N, ag_len, 5, 34)
    gbsa = data['gbsa'].reshape(-1, 1)  # shape: (N, 1)

    # Scale continuous features: x, y, z at indices 0:3
    continuous_idx = slice(0, 3)
    ab_cont = ab[..., continuous_idx].reshape(-1, 3)
    ag_cont = ag[..., continuous_idx].reshape(-1, 3)

    ab[..., continuous_idx] = feature_scaler.transform(ab_cont).reshape(ab.shape[0], ab.shape[1], ab.shape[2], 3)
    ag[..., continuous_idx] = feature_scaler.transform(ag_cont).reshape(ag.shape[0], ag.shape[1], ag.shape[2], 3)

    # Predict using model
    predictions_scaled = model.predict({'ab_input': ab, 'ag_input': ag}, verbose=1)

    # Inverse scale predictions and true labels
    predictions = label_scaler.inverse_transform(predictions_scaled)

    # Compute difference
    differences = predictions.flatten() - gbsa.flatten()

    # Print header
    print("-" * 80)
    print(f"{'Index':<10} {'Predicted':<15} {'Real':<15} {'Difference':<15}")
    print("-" * 80)

    for i in range(len(predictions)):
        print(f"{i:<10} {predictions[i][0]:<15.4f} {gbsa[i]} {differences[i]:<15.4f}")

    # Evaluation metrics
    mae = mean_absolute_error(gbsa, predictions)
    mse = mean_squared_error(gbsa, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(gbsa, predictions)

    print("\n📊 Evaluation on Training Data (Mock Test):")
    print(f"MAE:   {mae:.4f}")
    print(f"MSE:   {mse:.4f}")
    print(f"RMSE:  {rmse:.4f}")
    print(f"R²:    {r2:.4f}")
    print(f"Mean Δ: {np.mean(differences):.4f}")
    print(f"Std Δ:  {np.std(differences):.4f}")

    return {
        'predictions': predictions.flatten(),
        'real_values': gbsa.flatten(),
        'differences': differences,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }
