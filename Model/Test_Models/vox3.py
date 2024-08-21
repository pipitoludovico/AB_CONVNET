import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from scipy.ndimage import rotate

num_samples = 2000

protein_data = [np.random.random((np.random.randint(0, 51), 2, 5, 3)) for _ in range(num_samples)]

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

labels = np.ones(shape=adjusted_protein_data_np.shape[0])
for idx, data in enumerate(adjusted_protein_data_np):
    if (data == 0).all():
        labels[idx] = 0
print(labels)
num_arrays_with_zero_dim = sum(1 for array_ in protein_data if array_.shape[0] == 0)
print(f'Number of arrays that started with a dimension of 0: {num_arrays_with_zero_dim}')

# Split the data into training (70%), validation (20%), and testing (10%) sets
X_train, X_temp, y_train, y_temp = train_test_split(adjusted_protein_data_np, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1 / 3, random_state=42)


def coordinates_to_voxel_grid(coords, grid_size_=32, voxel_size=1.0):
    """
    Converts a set of coordinates into a voxel grid.

    Parameters:
    coords (ndarray): Array of shape (N, 2, 5, 3) representing pairs of 3D coordinates.
    grid_size (int): Size of the voxel grid along each dimension.
    voxel_size (float): Size of each voxel.

    Returns:
    ndarray: Voxel grid of shape (grid_size, grid_size, grid_size).
    """
    voxel_grid = np.zeros((grid_size_, grid_size_, grid_size_), dtype=np.float32)
    for pair in coords:
        for atom in pair:
            for coord in atom:
                x, y, z = coord
                # Map coordinates to voxel grid indices
                xi = int((x + grid_size_ / 2) // voxel_size)
                yi = int((y + grid_size_ / 2) // voxel_size)
                zi = int((z + grid_size_ / 2) // voxel_size)
                if 0 <= xi < grid_size_ and 0 <= yi < grid_size_ and 0 <= zi < grid_size_:
                    voxel_grid[xi, yi, zi] = 1.0
    return voxel_grid


grid_size = 32  # Define the grid size for the voxel grid

# Convert the training, validation, and test sets to voxel grids
X_train_voxelized = np.array([coordinates_to_voxel_grid(pairs, grid_size) for pairs in X_train])
X_val_voxelized = np.array([coordinates_to_voxel_grid(pairs, grid_size) for pairs in X_val])
X_test_voxelized = np.array([coordinates_to_voxel_grid(pairs, grid_size) for pairs in X_test])

# Reshape to add channel dimension (required by Keras Conv3D)
X_train_voxelized = X_train_voxelized[..., np.newaxis]
X_val_voxelized = X_val_voxelized[..., np.newaxis]
X_test_voxelized = X_test_voxelized[..., np.newaxis]


def augment_voxel_grid(voxel_grid):
    # Random rotation
    angle = np.random.uniform(-10, 10)  # Rotation angle in degrees
    axes = np.random.uniform(0, 1)  # Choose axes to rotate around
    voxel_grid = rotate(voxel_grid, angle, axes=(0, 1), reshape=False)

    # Random translation
    shift = np.random.uniform(-2, 2, size=3)  # Shift in each dimension
    voxel_grid = np.roll(voxel_grid, shift.astype(int), axis=(0, 1, 2))

    # Random flip
    if np.random.random() < 0.5:
        voxel_grid = np.flip(voxel_grid, axis=0)
    if np.random.random() < 0.5:
        voxel_grid = np.flip(voxel_grid, axis=1)
    if np.random.random() < 0.5:
        voxel_grid = np.flip(voxel_grid, axis=2)
    print("VOXEL AUG GRID SHAPE: ", voxel_grid.shape)
    return voxel_grid


# Apply augmentation to the training set
X_train_augmented = np.array([augment_voxel_grid(grid) for grid in X_train_voxelized])
X_val_augmented = np.array([augment_voxel_grid(grid) for grid in X_val_voxelized])
X_test_augmented = np.array([augment_voxel_grid(grid) for grid in X_test_voxelized])


def get_model(width=32, height=32, depth=32):
    """Build a 3D convolutional neural network model."""

    inputs = tf.keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu", padding="same")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    model_ = tf.keras.Model(inputs, outputs, name="3dcnn")
    return model_


model = get_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
# history = model.fit(X_train_voxelized, y_train, epochs=20, batch_size=32, validation_data=(X_val_voxelized, y_val))
history = model.fit(X_train_augmented, y_train, epochs=20, batch_size=32, validation_data=(X_val_augmented, y_val))

model.summary()
