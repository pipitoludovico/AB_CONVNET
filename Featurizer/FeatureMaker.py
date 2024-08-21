from os import cpu_count, chdir, makedirs
from multiprocessing import Pool, Manager
import tensorflow as tf

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

features = ['GBSA', 'NumContacts', 'NumPatches']


class FeaturizerClass:
    def __init__(self, dbDict: dict, root: str):
        self.dbDict = dbDict
        self.root = root
        self.datFile = "FINAL_DECOMP_MMPBSA.dat"
        self.dataframe = pd.DataFrame(columns=[*features], dtype=float)
        self.couples = {}

    def Featurize(self, pdbID: str, shared_list: list, shared_couples: dict) -> None:
        try:
            chdir("selected/" + pdbID)
            dec_res = self.GetDecomp()
            df: pd.DataFrame = self.dataframe.copy()
            shared_dic = self.update_dataframe(df, pdbID, dec_res)
            shared_couples[pdbID] = shared_dic[pdbID]  # Update the shared dictionary
            shared_list.append(df)  # Append modified dataframe to the shared list
            logging.debug(f"Appended DataFrame for {pdbID}")
        except Exception as e:
            logging.error(f"Error processing {pdbID}: {e}")
        finally:
            chdir(self.root)

    def GetDecomp(self) -> (list, list):
        dec_results = {}

        def WriteTotalFromDecomp():
            decomp_purged = open("total_purged.csv", 'w')
            with open(f'{self.datFile}', 'r') as f:
                for purgedLine in f.readlines():
                    decomp_purged.writelines(purgedLine)
                    if purgedLine == "\n" or purgedLine == " ":
                        break

        def GetResults():
            with open('total_purged.csv', 'r') as f:
                after_header = f.readlines()[7:]
                for lines in after_header:
                    if lines == "" or lines == " " or lines == "\n":
                        continue
                    if lines.split()[1].split(',')[1] == "R" or lines.split()[1].split(',')[1] == 'L':
                        resname = lines[0:4].strip()
                        resnum = lines[4:7].strip()
                        total_energy = float(lines.split(",")[-3])
                        if (resname + ":" + resnum) not in dec_results:
                            dec_results[resname + ":" + resnum] = total_energy

        WriteTotalFromDecomp()
        GetResults()
        return dec_results

    @staticmethod
    def update_dataframe(df_, pdbID: str, dec_res_: dict) -> dict:
        interfaceRes = []
        coordinates = {}
        with open('patches/selection_i0.pdb', 'r') as interfaceFile:
            for line in interfaceFile.readlines():
                if len(line.split()) > 2:
                    if float(line.split()[10]) > 0.15:
                        residueID = line.split()[3] + ":" + line[23:27].strip()
                        if residueID not in interfaceRes:
                            interfaceRes.append(residueID)

        df_.at[pdbID, 'Nprotein_data.numPatches'] = len(interfaceRes)

        for resname, energy in dec_res_.items():
            if resname in interfaceRes:
                df_.at[pdbID, resname] = energy

        with open('results_mmgbsa.dat', 'r') as GBSA:
            for line in GBSA.readlines():
                if "DELTA TOTAL" in line:
                    GBSA = float(line.split()[2])
                    df_.at[pdbID, 'GBSA'] = GBSA
                    break

        with open('complex_minimized_chains.pdb', 'r') as complexPDB:
            for line in complexPDB.readlines():
                if len(line.split()) > 4:
                    resname_pdb = line[17:21].strip()
                    resid_pdb = line[22:27].strip()
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    key = resname_pdb + ":" + resid_pdb
                    if key not in coordinates.keys():
                        coordinates[key.replace("\n", "")] = []
                    coordinates[key.replace("\n", '')].append(np.array([x, y, z]))

        for key in coordinates.keys():
            coordinates[key] = np.vstack(coordinates[key])
        shared_couples = {pdbID: []}
        with open('contacts.int', 'r') as contacts:
            contact_content = contacts.read()
            results = contact_content.replace("\n", '').split(',')[1:]
            numContacts = contact_content.split(",")[0]
            df_.at[pdbID, 'NumContacts'] = numContacts
            for result in results:
                lig_res = result.split("-")[0]
                rec_res = result.split("-")[1]

                lig_energy = np.full((coordinates[lig_res].shape[0], 1), dec_res_[lig_res])
                rec_energy = np.full((coordinates[rec_res].shape[0], 1), dec_res_[rec_res])

                lig_score_stack = np.hstack((coordinates[lig_res], lig_energy))
                rec_score_stack = np.hstack((coordinates[rec_res], rec_energy))
                # print(np.reshape(lig_score_stack, -1)) # diventa 1D
                # print(lig_score_stack) # [[]] x numbero di coppie => ok?
                # print(np.vstack((lig_score_stack, rec_score_stack))) # probabilmente è questo...

                # shared_couples.append([np.reshape(lig_score_stack, -1), np.reshape(rec_score_stack, -1)])
                shared_couples[pdbID].append(np.vstack((lig_score_stack, rec_score_stack)))
        return shared_couples

    def ParallelFeaturize(self) -> None:
        cpuUnits = int(cpu_count() // 4)
        manager = Manager()
        shared_list = manager.list()  # Create a shared list
        shared_couples = manager.dict()
        with Pool(processes=cpuUnits) as p:
            for pdbID, chains in self.dbDict.items():
                logging.debug(f"Starting Featurize for {pdbID}")
                p.apply_async(self.Featurize, args=(pdbID, shared_list, shared_couples,))
            p.close()
            p.join()

        logging.debug("All processes have completed. Concatenating DataFrames.")
        for df in shared_list:
            self.dataframe = pd.concat([self.dataframe, df])
        self.dataframe.fillna(0, inplace=True)
        logging.debug("Final DataFrame concatenated and filled with NaNs replaced by 0.")
        self.couples = dict(shared_couples)
        del shared_couples
        logging.debug("Adding couples to the final array.")

    def GetDataAsDataframe(self) -> pd.DataFrame:
        return self.dataframe

    def GetCouples(self):
        max_lengths = [max(array.shape[0] for array in coords) for coords in self.couples.values()]

        num_samples = len(self.couples)
        max_length = max(max_lengths)
        num_features = 4  # this is how I want the tensor: x, y, z, (gbsa for that receptor)

        protein_data = np.zeros((num_samples, max_length, num_features), dtype=np.float32)

        for i, (pdb, coords) in enumerate(self.couples.items()):
            for j, array in enumerate(coords):
                protein_data[i, :array.shape[0], :] = array

        protein_data_tensor = tf.convert_to_tensor(protein_data)
        dataset = tf.data.Dataset.from_tensor_slices(protein_data_tensor)
        # Save the dataset
        makedirs('saved_results')
        tf.data.experimental.save(dataset, './saved_results/protein_data')

        tensor_string = tf.io.serialize_tensor(protein_data_tensor)
        tf.io.write_file('./saved_results/protein_data.tfrecord', tensor_string)

        np.save('./saved_results/protein_data.npy', protein_data)
        return protein_data_tensor
