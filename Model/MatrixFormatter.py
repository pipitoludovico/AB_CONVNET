import os.path
from os import listdir, makedirs, path, getcwd, chdir
import numpy as np

samples = []


def FormatData():
    def LoadData():
        cwd = getcwd()
        chdir(cwd)
        if not path.exists("matrices"):
            makedirs('matrices', exist_ok=True)
        for pdbFolder in listdir("selected"):
            if os.path.exists("selected/" + pdbFolder + "/saved_results/" + "protein_data_noDEC.npy"):
                sample = np.load(("selected/" + pdbFolder + "/saved_results/" + "protein_data_noDEC.npy"), allow_pickle=True)
                if sample.shape[0] == 5201:
                    samples.append(sample)

    # def PadData():
    #     max_len = max([arr.shape[0] for arr in samples])
    #     print("Max length:", max_len)
    #     padded = [np.pad(arr, ((0, max_len - arr.shape[0]), (0, 0)), 'constant', constant_values=0) if arr.shape[0] < max_len else arr for arr in samples]
    #     for x, y in zip(padded, samples):
    #         assert x.shape[0] == max_len, f"Padding failed: {x.shape[0]} != {max_len}"
    #     return padded

    def SavePadded():
        # padded = PadData()
        np.save('matrices/padded.npy', np.vstack(samples), allow_pickle=True)

    LoadData()
    SavePadded()
