from os import walk, path
import numpy as np
import joblib
from keras.models import load_model
from keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def Test(args):
    """
    Test the model using samples from the './predictions' folder,
    compare predicted values with actual values from a comparison file,
    and print results along with evaluation metrics.
    """
    # Load the feature scaler and label scaler
    try:
        feature_scaler = joblib.load('feature_scaler.pkl')
        label_scaler = joblib.load('label_scaler.pkl')
        print("Feature scaler and label scaler loaded successfully.")
    except Exception as e:
        print(f"Error loading scalers: {e}")
        return None

    # Load the model
    try:
        model = load_model(args['model'], compile=False)  # Load the model without compiling
        model.compile(optimizer=Adam(learning_rate=args['lr']),
                      loss="mean_absolute_error",
                      metrics=["mean_absolute_error"])
        print("Model loaded and compiled successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # Initialize lists to store results
    predictions = []
    actual_values = []
    sample_names = []
    differences = []

    print("\nPredictions for each sample:")
    print("-" * 80)
    print(f"{'Sample Name':<20} {'Predicted':<15} {'Real':<15} {'Diff':<15}")
    print("-" * 80)

    # Walk through the predictions folder
    for folder, _, samples in walk('./predictions'):
        for file in samples:
            if not file.endswith('npy'):
                continue

            name = file.split(".")[0]
            sample_path = path.join(folder, file)

            try:
                # Load the sample data
                sample_data = np.load(sample_path)

                # Ensure consistent preprocessing
                if sample_data.ndim == 1:  # Reshape if it's 1D
                    sample_data = sample_data.reshape(1, -1)

                if sample_data.shape[1] < 1700:
                    padding = np.zeros((sample_data.shape[0], 1700 - sample_data.shape[1]))
                    sample_data = np.hstack((sample_data, padding))

                # Scale the features
                X_scaled = feature_scaler.transform(sample_data)
                X_scaled = X_scaled[:, :, np.newaxis]  # Reshape for Conv1D model

                # Generate predictions
                prediction_scaled = model.predict(X_scaled, verbose=0)

                # Inverse scale the predictions
                prediction_unscaled = label_scaler.inverse_transform(prediction_scaled.reshape(-1, 1))[0][0]

                # Get the real value from the comparison file
                with open('comparazione', 'r') as comp_file:
                    real_value = next((float(line.split()[3]) for line in comp_file if name in line), None)

                if real_value is not None:
                    # Calculate the difference between predicted and actual values
                    diff = prediction_unscaled - real_value

                    # Store results
                    predictions.append(prediction_unscaled)
                    actual_values.append(real_value)
                    sample_names.append(name)
                    differences.append(diff)

                    # Print results for this sample
                    print(f"{name:<20} {prediction_unscaled:< 15.4f} {real_value:< 15.4f} {diff:< 15.4f}")

            except Exception as e:
                print(f"Error processing {name}: {e}")
                continue

    print("-" * 80)

    # Calculate metrics if predictions are available
    if len(predictions) > 0:
        predictions = np.array(predictions)
        actual_values = np.array(actual_values)
        differences = np.array(differences)

        mae = np.mean(np.abs(differences))
        mse = np.mean(differences ** 2)
        rmse = np.sqrt(mse)
        r2 = 1 - (np.sum((actual_values - predictions) ** 2) / np.sum((actual_values - np.mean(actual_values)) ** 2))

        print("\nTest Set Metrics:")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R² Score: {r2:.4f}")

        return {
            'predictions': predictions,
            'actual_values': actual_values,
            'sample_names': sample_names,
            'differences': differences,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
    else:
        print("\nNo predictions were successfully generated.")
        return None


def Test2(args):
    """
    Testing function that loads the entire training dataset and evaluates the model on it.
    Applies proper scaling and compares with actual values from comparison file.
    """
    # Load the model
    try:
        model = load_model(args['model'])
        print(f"Successfully loaded model from {args['model']}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # Initialize lists to store results
    predictions = []
    actual_values = []
    sample_names = []
    differences = []

    print("\nPredictions for each sample:")
    print("-" * 80)
    print(f"{'Sample Name':<20} {'Original Shape':<20} {'Predicted':<15} {'Real':<15} {'Diff':<15}")
    print("-" * 80)

    # Load the scalers (ensure these are saved properly after training)
    feature_scaler = joblib.load("feature_scaler.pkl")
    label_scaler = joblib.load("label_scaler.pkl")

    # Load the training dataset
    dataset = np.load("./matrices/padded.npy", allow_pickle=True)
    X = dataset[:, :-1]  # Features (2D: samples x features)
    y = dataset[:, -1]  # Labels (1D: samples)

    # Reshape and pad if necessary
    X_resized = X.copy()  # You may need to modify the reshaping depending on your data
    if X_resized.shape[1] < 2600:
        padding = np.zeros((X_resized.shape[0], 2600 - X_resized.shape[1]))
        X_resized = np.hstack((X_resized, padding))

    # Scale features using the saved feature scaler
    X_scaled = feature_scaler.transform(X_resized)  # Use transform, not fit_transform
    X_scaled = X_scaled[:, :, np.newaxis]  # Reshape for Conv1D model

    # Generate predictions for the entire dataset
    predictions_scaled = model.predict(X_scaled, verbose=0)

    # Inverse scale the predictions
    predictions_unscaled = label_scaler.inverse_transform(predictions_scaled.reshape(-1, 1))

    # Get the real values (from dataset)
    real_values = y

    for idx, name in enumerate(dataset[:, 0]):  # Assuming the name is in the first column of the dataset
        prediction_unscaled = predictions_unscaled[idx][0]
        real = real_values[idx]

        # Calculate the difference between the predicted and real values
        diff = prediction_unscaled - real

        # Store the results
        predictions.append(prediction_unscaled)
        actual_values.append(real)
        sample_names.append(idx)
        differences.append(diff)

        # Print detailed output
        print(f"{idx:<20} {str(X_resized.shape):<20} {prediction_unscaled:< 15.4f} "
              f"{real:< 15.4f} {diff:< 15.4f}")

    print("-" * 80)

    # Only calculate metrics if we have predictions
    if len(predictions) > 0:
        # Convert lists to numpy arrays
        predictions = np.array(predictions)
        actual_values = np.array(actual_values)
        differences = np.array(differences)

        # Calculate metrics
        mae = mean_absolute_error(actual_values, predictions)
        mse = mean_squared_error(actual_values, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual_values, predictions)

        # Print metrics
        print("\nTest Set Metrics:")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"Mean Difference: {np.mean(differences):.4f}")
        print(f"Std Dev of Differences: {np.std(differences):.4f}")

        return {
            'predictions': predictions,
            'actual_values': actual_values,
            'sample_names': sample_names,
            'differences': differences,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
    else:
        print("\nNo predictions were successfully generated.")
        return None
