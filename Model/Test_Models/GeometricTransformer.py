import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Generate 200 random protein pairs
num_samples = 2000
max_pairs = 5  # The maximum number of pairs you want in your dataset

protein_data = [np.random.random((np.random.randint(1, max_pairs+1), 2, 5, 3)) for _ in range(num_samples)]

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
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)

# Convert labels to one-hot encoding
y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_val_one_hot = tf.keras.utils.to_categorical(y_val, num_classes=2)
y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes=2)

# Define a simple 3D CNN model with adjusted pooling
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(max_pairs_in_dataset, 2, 5, 3)),
    tf.keras.layers.Conv3D(32, (3, 2, 2), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling3D(pool_size=(2, 1, 1)),
    tf.keras.layers.Conv3D(64, (3, 2, 2), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling3D(pool_size=(2, 1, 1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Define the optimizer with a custom learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Compile the model with the optimizer and the binary_crossentropy loss function
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with validation data
model.fit(X_train, y_train_one_hot, epochs=1000, validation_data=(X_val, y_val_one_hot))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test_one_hot)
print(f"Test accuracy: {test_accuracy}")

# Print the model summary
model.summary()
