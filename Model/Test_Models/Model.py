import os

import numpy as np
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if logs.get('accuracy') >= 0.85 and logs.get('loss') < 0.3:
            print("\nReached 85% accuracy and <0.3 loss, so cancelling training!")
            self.model.stop_training = True


callbacks = myCallback()

sampl = np.random.uniform(low=-10, high=10, size=(5000, 222))
new_column = np.random.choice([0, 1], size=(5000, 1)).astype(int)
sampl = np.column_stack((sampl, new_column))

training = sampl[:3200]
test = sampl[3200:]

training_features = training[:, :-1]
training_labels = training[:, -1]

test_features = test[:, :-1]
test_labels = test[:, -1]

if not os.path.exists('model_ludo.h5'):

    # import tensorflow as tf
    # from tensorflow.keras.models import Sequential
    # from tensorflow.keras.layers import Dense, Conv1D, Flatten, Masking, TimeDistributed
    #
    # # Define the model
    # model = Sequential([
    #     Masking(mask_value=0.0, input_shape=(max_pairs, padded_protein_data.shape[2])),
    #     TimeDistributed(Conv1D(32, kernel_size=3, activation='relu')),
    #     TimeDistributed(Flatten()),
    #     TimeDistributed(Dense(64, activation='relu')),
    #     TimeDistributed(Dense(1, activation='linear'))
    # ])
    #
    # # Compile the model
    # model.compile(optimizer='adam', loss='mean_squared_error')
    #
    # # Train the model
    # model.fit(padded_protein_data, padded_scores_data, epochs=10, batch_size=1)
    #
    # # Print the model summary
    # model.summary()

    model = tf.keras.models.Sequential([
        tf.keras.layers.Reshape((222, 1), input_shape=(222,)),  # Reshape for compatibility with 1D convolution
        tf.keras.layers.Conv1D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.kera.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid'),  # Sigmoid activation for binary classification
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(training_features, training_labels, epochs=30, callbacks=[callbacks])
    model.evaluate(test_features, test_labels)
    model.save('model_ludo.h5')
else:
    loaded_model = tf.keras.models.load_model('model_ludo.h5')
    classifications = loaded_model.predict(test_features)
    new_data = np.random.uniform(low=-10, high=10, size=(1, 222))
    predictions = loaded_model.predict(new_data)
    for i in range(10):
        print(
            f"Example {i + 1}: True Label: {test_labels[i]}, Raw Probability: {classifications[i][0]}, Antigen Binder?: {bool(classifications[i][0] > 0.5)}")
    print(predictions)
