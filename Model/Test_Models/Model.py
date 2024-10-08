import os

import numpy as np
import tensorflow as tf
from keras.callbacks import Callback
from keras.models import Sequential
from keras import layers
np.random.seed(42)
tf.random.set_seed(42)


class myCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if logs.get('accuracy') >= 0.85 and logs.get('loss') < 0.3:
            print("\nReached 85% accuracy and <0.3 loss, so cancelling training!")
            self.model.stop_training = True


callbacks = myCallback()

if not os.path.exists('model_ludo.h5'):
    model = Sequential([
        layers.Reshape((222, 1), input_shape=(222,)),  # Reshape for compatibility with 1D convolution
        layers.Conv1D(32, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid'),  # Sigmoid activation for binary classification
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
