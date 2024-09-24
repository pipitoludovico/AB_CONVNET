import os
import sys
import numpy as np

np.set_printoptions(threshold=sys.maxsize)


# import tensorflow as tf
# from keras import layers, models
# from sklearn.model_selection import train_test_split

def LoadData():
    samples = []

    for pdbFolder in os.listdir('selected'):
        for array in os.listdir("selected/" + pdbFolder + "/saved_results"):
            if array.endswith('npy'):
                sample = np.load(("selected/" + pdbFolder + "/saved_results/" + array), allow_pickle=True)
                samples.append(sample)

    # print("Loading from experimental:")
    # loaded_dataset = tf.data.Dataset.load('../saved_results/protein_data')
    data_arrays = []
    labels = []
    for sample in samples:
        print(sample.shape)
        print(sample)

    print(labels)
# print(data_arrays[0])
# loaded_array = np.array(data_arrays)
# loaded_labels = np.array(labels)
#
# Generate labels (assuming your dataset contains labels)
# labels = loaded_array[3]  # Example labels, replace with actual labels
# Split the data into training (70%), validation (20%), and testing (10%) sets
# X_train, X_test, y_train, y_test = train_test_split(loaded_array, labels, test_size=0.3, random_state=42)

# print(labels)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# print(X_train[0], X_test[0], y_train[0], y_test[0])
