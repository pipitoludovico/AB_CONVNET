import os

import keras.models
import numpy as np
from sys import maxsize

np.set_printoptions(threshold=maxsize)


def Test(data):
    model = None
    with open('scaling_factor', 'r') as sf:
        scaling_factor = float(sf.readline())
    for model in os.listdir('.'):
        if model.endswith(".keras"):
            model = keras.models.load_model(model)
    for samples in data:
        sample = np.array(samples)
        prediction = model.predict(sample)
        prediction_rescaled = prediction * scaling_factor
        print(prediction_rescaled)
