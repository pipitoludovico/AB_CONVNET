import os
import numpy as np
import json


def FormatData():
    def pad_matrix(mat, target_residues):
        pad_len = target_residues - mat.shape[0]
        if pad_len <= 0:
            return mat
        padding = np.zeros((pad_len, 5, 34), dtype=mat.dtype)
        return np.vstack([mat, padding])

    def load_pad_config():
        with open("matrices/pad_config.json", "r") as f:
            config = json.load(f)
        return config["max_ab_len"], config["max_ag_len"]

    def LoadData():
        os.makedirs('matrices', exist_ok=True)
        base_path = "./selected"
        samples = []

        # Compute max lengths
        max_ab_len = 0
        max_ag_len = 0
        json_path = "matrices/pad_config.json"
        if not os.path.exists(json_path):
            for pdbFolder in os.listdir(base_path):
                saved_results_path = os.path.join(base_path, pdbFolder, "saved_results")
                if not os.path.isdir(saved_results_path):
                    continue

                for file in os.listdir(saved_results_path):
                    file_path = os.path.join(saved_results_path, file)
                    if file.endswith(".npy"):
                        try:
                            data = np.load(file_path)
                            if 'abMatrix' in file:
                                max_ab_len = max(max_ab_len, data.shape[0])
                            elif 'agMatrix' in file:
                                max_ag_len = max(max_ag_len, data.shape[0])
                        except Exception as e:
                            print(f"Failed to load {file_path}: {e}")
            with open("matrices/pad_config.json", "w") as f:
                json.dump({"max_ab_len": max_ab_len, "max_ag_len": max_ag_len}, f)
        else:
            max_ab_len, max_ag_len = load_pad_config()

        # Load data
        for pdbFolder in os.listdir(base_path):
            folder_path = os.path.join(base_path, pdbFolder, "saved_results")
            if not os.path.isdir(folder_path):
                continue
            ab_path = None
            ag_path = None
            label_path = None
            # carico le path
            for file in os.listdir(folder_path):
                if "abMatrix" in file and file.endswith(".npy"):
                    ab_path = os.path.join(folder_path, file)
                elif "agMatrix" in file and file.endswith(".npy"):
                    ag_path = os.path.join(folder_path, file)
                elif file == "label.npy":
                    label_path = os.path.join(folder_path, file)
                # qui carico gli array...
                if ab_path and ag_path and label_path:
                    ab = np.load(ab_path)
                    ag = np.load(ag_path)
                    gbsa = np.load(label_path)
                    ab_padded = pad_matrix(ab, max_ab_len)
                    ag_padded = pad_matrix(ag, max_ag_len)
                    samples.append((ab_padded, ag_padded, gbsa))
        return samples

    def remove_outliers_by_label(samples_, lower_percentile=5, upper_percentile=95):
        labels = [sample[2] for sample in samples_]
        labels_array = np.array(labels)
        lower_bound = np.percentile(labels_array, lower_percentile)
        upper_bound = np.percentile(labels_array, upper_percentile)
        print(f"Filtering labels using percentiles: Values outside [{lower_bound:.2f}, {upper_bound:.2f}]")

        filtered = [sample for sample in samples_ if lower_bound <= sample[2] <= upper_bound]
        print(f"Filtered dataset size: {len(filtered)} (removed {len(samples_) - len(filtered)} samples)")
        return filtered

    def SavePadded(samples, path='matrices/padded_dataset.npz'):
        ab_all = []
        ag_all = []
        gbsa_all = []

        for ab, ag, gbsa in samples:
            ab_all.append(ab)
            ag_all.append(ag)
            gbsa_all.append(gbsa)

        np.savez_compressed(
            path,
            ab=np.array(ab_all, dtype=np.float32),
            ag=np.array(ag_all, dtype=np.float32),
            gbsa=np.array(gbsa_all, dtype=np.float32)
        )
        print(f"Saved preprocessed data to: {path}")

    samples_list = LoadData()
    filtered_samples = remove_outliers_by_label(samples_list, lower_percentile=1, upper_percentile=99)
    np.random.shuffle(filtered_samples)  # val loss improves a lot
    SavePadded(samples_list)
