import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Generate 2000 random protein pairs
num_samples = 2000
max_pairs = 5  # The maximum number of pairs you want in your dataset

protein_data = [np.random.random((np.random.randint(1, max_pairs + 1), 2, 5, 3)) for _ in range(num_samples)]

# Find the maximum number of pairs in the dataset
max_pairs_in_dataset = max(x.shape[0] for x in protein_data)

# Adjust each array to have the same number of pairs by padding with zeros
adjusted_protein_data = []
for array_ in protein_data:
    if array_.shape[0] < max_pairs_in_dataset:
        num_pairs_to_add = max_pairs_in_dataset - array_.shape[0]
        added = np.zeros((num_pairs_to_add, array_.shape[1], array_.shape[2], array_.shape[3]))
        array_ = np.concatenate((array_, added), axis=0)
    adjusted_protein_data.append(array_)

adjusted_protein_data_np = np.array(adjusted_protein_data)

# Generate random labels for binary classification
labels = np.random.randint(0, 2, size=(num_samples,))

# Split the data into training (70%), validation (20%), and testing (10%) sets
X_train, X_temp, y_train, y_temp = train_test_split(adjusted_protein_data_np, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1 / 3, random_state=42)

# Convert labels to one-hot encoding
y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_val_one_hot = tf.keras.utils.to_categorical(y_val, num_classes=2)
y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes=2)


def coordinates_to_voxel_grid(coords, grid_size=32, voxel_size=1.0):
    """
    Converts a set of coordinates into a voxel grid.
    Parameters:
    coords (ndarray): Array of shape (N, 2, 5, 3) representing pairs of 3D coordinates.
    grid_size (int): Size of the voxel grid along each dimension.
    voxel_size (float): Size of each voxel.
    Returns:
    ndarray: Voxel grid of shape (grid_size, grid_size, grid_size).
    """
    voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    for pair in coords:
        for atom in pair:
            for coord in atom:
                x, y, z = coord
                # Map coordinates to voxel grid indices
                xi = int((x + grid_size / 2) // voxel_size)
                yi = int((y + grid_size / 2) // voxel_size)
                zi = int((z + grid_size / 2) // voxel_size)
                if 0 <= xi < grid_size and 0 <= yi < grid_size and 0 <= zi < grid_size:
                    voxel_grid[xi, yi, zi] = 1.0
    return voxel_grid


grid_size = 32  # Define the grid size for the voxel grid


# Convert the entire dataset to voxel grids
def convert_dataset_to_voxels(data, grid_size):
    return np.array([coordinates_to_voxel_grid(pairs, grid_size) for pairs in data])


X_train_voxelized = convert_dataset_to_voxels(X_train, grid_size)
X_val_voxelized = convert_dataset_to_voxels(X_val, grid_size)
X_test_voxelized = convert_dataset_to_voxels(X_test, grid_size)

# Reshape to add channel dimension (required by Keras Conv3D)
X_train_voxelized = X_train_voxelized[..., np.newaxis]
X_val_voxelized = X_val_voxelized[..., np.newaxis]
X_test_voxelized = X_test_voxelized[..., np.newaxis]

# Adjust the shape to (num_samples, max_pairs, grid_size, grid_size, grid_size, 1)
X_train_voxelized = X_train_voxelized.reshape(X_train_voxelized.shape[0], max_pairs_in_dataset, grid_size, grid_size,
                                              grid_size, 1)
X_val_voxelized = X_val_voxelized.reshape(X_val_voxelized.shape[0], max_pairs_in_dataset, grid_size, grid_size,
                                          grid_size, 1)
X_test_voxelized = X_test_voxelized.reshape(X_test_voxelized.shape[0], max_pairs_in_dataset, grid_size, grid_size,
                                            grid_size, 1)


def create_3D_convnet(input_shape):
    model = models.Sequential()
    model.add(layers.TimeDistributed(layers.Conv3D(32, (3, 3, 3), activation='relu'), input_shape=input_shape))
    model.add(layers.TimeDistributed(layers.MaxPooling3D((2, 2, 2))))
    model.add(layers.TimeDistributed(layers.Conv3D(64, (3, 3, 3), activation='relu')))
    model.add(layers.TimeDistributed(layers.MaxPooling3D((2, 2, 2))))
    model.add(layers.TimeDistributed(layers.Conv3D(128, (3, 3, 3), activation='relu')))
    model.add(layers.TimeDistributed(layers.MaxPooling3D((2, 2, 2))))
    model.add(layers.TimeDistributed(layers.Flatten()))
    model.add(layers.TimeDistributed(layers.Dense(128, activation='relu')))
    model.add(layers.TimeDistributed(layers.Dense(1, activation='linear')))
    model.add(layers.Flatten())
    model.add(layers.Dense(2, activation='softmax'))  # Assuming binary classification
    return model


input_shape = (max_pairs_in_dataset, grid_size, grid_size, grid_size, 1)  # Shape of each residue pair voxel grid
model = create_3D_convnet(input_shape)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train_voxelized, y_train_one_hot, epochs=20, batch_size=32,
                    validation_data=(X_val_voxelized, y_val_one_hot))
