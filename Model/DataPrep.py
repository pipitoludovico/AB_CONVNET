import numpy as np
from sklearn.preprocessing import StandardScaler


def PrepareData(dataset_path):
    """Prepare and scale the data for a Conv1D network, including scaling the labels."""

    # Load dataset
    dataset = np.load(dataset_path, allow_pickle=True)

    # Separate features and labels
    X = dataset[:, :-1]
    y = dataset[:, -1].reshape(-1, 1)  # Reshape to 2D for scaler

    # Scale features and labels separately
    feature_scaler = StandardScaler()
    label_scaler = StandardScaler()

    X_scaled = feature_scaler.fit_transform(X)
    y_scaled = label_scaler.fit_transform(y)

    # Combine scaled features and labels
    X_scaled = np.hstack([X_scaled, y_scaled])

    return X_scaled, feature_scaler, label_scaler
