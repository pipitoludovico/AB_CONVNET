import numpy as np
from sklearn.preprocessing import StandardScaler


# def PrepareData(dataset_path):
#     """Prepare and scale the data for a Conv1D network, including scaling the labels if necessary."""
#     # Load dataset
#     dataset = np.load(dataset_path, allow_pickle=True)
#     X = dataset[:, :-1]  # Features (2D: samples x features)
#     y = dataset[:, -1]  # Labels (1D: samples)
#
#     # Create and fit the scaler for features (scale only 2D data)
#     feature_scaler = StandardScaler()
#     X_scaled = feature_scaler.fit_transform(X)  # Scale features (2D: samples x features)
#
#     # Optionally scale labels (for regression tasks)
#     label_scaler = StandardScaler()
#     y_scaled = label_scaler.fit_transform(y.reshape(-1, 1))  # Reshape to 2D for scaler (samples, 1)
#
#     # Add a new axis to match Conv1D input shape (samples, timesteps, channels)
#     X_scaled = X_scaled[:, :, np.newaxis]  # Now 3D: (samples, timesteps, channels)
#
#     print("Shape after scaling and reshaping:", X_scaled.shape)
#
#     return X_scaled, y_scaled, feature_scaler, label_scaler

def PrepareData(dataset_path):
    """Prepare and scale the data for a Conv1D network, including scaling the labels if necessary."""
    # Load dataset
    dataset = np.load(dataset_path, allow_pickle=True)

    feature_scaler = StandardScaler()
    X_scaled = feature_scaler.fit_transform(dataset)

    return X_scaled, None, feature_scaler, None
