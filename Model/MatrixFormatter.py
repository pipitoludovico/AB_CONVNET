import os
import numpy as np
import pandas as pd


def FormatData():
    samples = []

    def LoadData():
        cwd = os.getcwd()
        os.chdir(cwd)
        numberOfPairs: int = 50

        if not os.path.exists("matrices"):
            os.makedirs('matrices', exist_ok=True)
        for pdbFolder in os.listdir("selected"):
            file_path = os.path.join("selected", pdbFolder, "saved_results", "protein_data_noDEC.npy")
            if os.path.exists(file_path):
                try:
                    sample = np.load(file_path, allow_pickle=True)
                    if sample.shape[0] == ((numberOfPairs * 60) + 1):  # Verify sample size
                        samples.append(sample)
                except Exception as e:
                    print(f"Error loading {pdbFolder}: {e}")
        print(f"Loaded {len(samples)} valid samples.")

    def remove_outliers_by_label(samples_, lower_percentile=15, upper_percentile=85):
        labels = [sample[-1] for sample in samples_]  # Estraggo le label (ultima colonna di ogni sample)
        labels_array = np.array(labels)
        # calcolo i percentili
        lower_bound = np.percentile(labels_array, lower_percentile)
        upper_bound = np.percentile(labels_array, upper_percentile)
        print(f"Filtering labels using percentiles: Values outside [{lower_bound:.2f}, {upper_bound:.2f}]")
        # Filtra i sample basandoti sui limiti delle GBSA
        filtered = [sample for sample in samples if lower_bound <= sample[-1] <= upper_bound]

        print(f"Filtered dataset size: {len(filtered)} (removed {len(samples_) - len(filtered)} samples)")
        return filtered

    def SavePadded(filtered_samples_):
        print("\nDataset description before saving:\n")
        df = pd.DataFrame(filtered_samples_)
        print(df.describe())
        np.save('matrices/padded.npy', filtered_samples_, allow_pickle=True)
        print("Dataset saved to 'matrices/padded.npy'")

    LoadData()
    samples_array = np.array(samples, dtype=float)

    # Rimuovi outlier basati sulle label
    filtered_samples = remove_outliers_by_label(samples_array, lower_percentile=0, upper_percentile=100)
    SavePadded(filtered_samples)
