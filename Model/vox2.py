import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

print("Loading from experimental:")
loaded_dataset = tf.data.experimental.load('../saved_results/protein_data')
data_arrays = []
for sample in loaded_dataset.as_numpy_iterator():
    data_arrays.append(sample)

loaded_array = np.array(data_arrays)

# Generate labels (assuming your dataset contains labels)
labels = np.ones(shape=(loaded_array.shape[0],))  # Example labels, replace with actual labels

# Split the data into training (70%), validation (20%), and testing (10%) sets
X_train, X_temp, y_train, y_temp = train_test_split(loaded_array, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1 / 3, random_state=42)

# Convert labels to one-hot encoding
y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_val_one_hot = tf.keras.utils.to_categorical(y_val, num_classes=2)
y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes=2)


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
    for coordsAndGBSA in coords:
        x, y, z = coordsAndGBSA[0], coordsAndGBSA[1], coordsAndGBSA[2]
        xi = int((x + grid_size_ / 2) // voxel_size)
        yi = int((y + grid_size_ / 2) // voxel_size)
        zi = int((z + grid_size_ / 2) // voxel_size)
        if 0 <= xi < grid_size_ and 0 <= yi < grid_size and 0 <= zi < grid_size:
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


def create_3D_convnet(input_shape_):
    model_ = models.Sequential()
    model_.add(layers.Conv3D(128, (3, 2, 2), activation='relu', input_shape=input_shape_, padding='same'), )
    model_.add(layers.MaxPooling3D((2, 1, 1)))
    model_.add(layers.Conv3D(128, (3, 2, 2), activation='relu'))
    model_.add(layers.MaxPooling3D((2, 2, 2)))
    model_.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))
    model_.add(layers.MaxPooling3D((2, 1, 1)))
    model_.add(layers.Flatten())
    model_.add(layers.Dense(128, activation='relu'))
    model_.add(layers.Dense(2, activation='softmax'))  # Assuming binary classification
    return model_


input_shape = (grid_size, grid_size, grid_size, 1)  # Shape of the voxel grid
model = create_3D_convnet(input_shape)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train_voxelized, y_train_one_hot, epochs=150, batch_size=64,
                    validation_data=(X_val_voxelized, y_val_one_hot))
model.save('model_ludo.keras')
