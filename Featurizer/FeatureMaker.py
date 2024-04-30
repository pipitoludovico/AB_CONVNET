import os.path
from os import cpu_count, chdir
from subprocess import run
from multiprocessing import Pool, Manager
import pandas as pd
from Interfacer.VMD import GetVDWcontacts
import itertools

aminoAcids = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
              'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']

receptorAAdecomp = ["r" + aa for aa in aminoAcids]
ligandAAdecomp = ["l" + aa for aa in aminoAcids]

combinations = list(itertools.combinations(aminoAcids, 2))

energyPairs = ["ene" + pair[0] + "-" + pair[1] for pair in combinations]
features = ['GBSA', 'NumContacts', 'NumPatches', 'PatchesContacts', ]


class FeaturizerClass:
    def __init__(self, dbDict: dict, root: str):
        self.dbDict = dbDict
        self.root = root
        self.datFile = "FINAL_DECOMP_MMPBSA.dat"
        self.dataframe = pd.DataFrame(
            columns=[*receptorAAdecomp, *ligandAAdecomp, *combinations, *energyPairs, *features], dtype=float)

    def Featurize(self, pdbID: str, chainsID: list, shared_list: list) -> None:
        chdir("selected/" + pdbID)
        if not os.path.exists('contacts.int'):
            GetVDWcontacts((chainsID[0], chainsID[1]), chainsID[2])
        rec_results, lig_results = self.GetDecomp()  # -> two dictionaries
        df: pd.DataFrame = self.dataframe.copy()
        self.update_dataframe(df, pdbID, rec_results, lig_results)
        shared_list.append(df)  # Append modified dataframe to the shared list
        chdir(self.root)

    @staticmethod
    def update_dataframe(df_, pdbID: str, rec_results: dict, lig_results: dict) -> None:
        for resname, energy in rec_results.items():
            col_name = "r" + resname
            if col_name in df_:
                df_.at[pdbID, col_name] = energy
        for resname, energy in lig_results.items():
            col_name = "l" + resname
            if col_name in df_.columns:
                df_.at[pdbID, col_name] = energy

        with open('results_mmgbsa.dat', 'r') as GBSA:
            for line in GBSA.readlines():
                if "DELTA TOTAL" in line:
                    GBSA = float(line.split()[2])
                    df_.at[pdbID, 'GBSA'] = GBSA
                    break
        with open('contacts.int', 'r') as NumContacts:
            for line in NumContacts.readlines():
                numContact = int(line.split(",")[0])
                df_.at[pdbID, 'NumContacts'] = numContact

                pairs = line.strip().split(',')[1:]
                for pair in pairs:
                    res1, res2 = pair.split("-")
                    label = (res1.strip(), res2.strip())
                    labelR = (res2.strip(), res1.strip())

                    df_.fillna(0, inplace=True)
                    if label in df_.columns:
                        df_.at[pdbID, label] = df_.at[pdbID, label] + 1
                    elif labelR in df_.columns:
                        df_.at[pdbID, labelR] = df_.at[pdbID, labelR] + 1

    def GetDecomp(self) -> (list, list):
        rec_decomp_results = {}
        lig_decomp_results = {}

        def CleanDecompHis():
            run(f"sed -i 's/HSP/HIS/g' {self.datFile}", shell=True)
            run(f"sed -i 's/HSE/HIS/g' {self.datFile}", shell=True)
            run(f"sed -i 's/HSD/HIS/g' {self.datFile}", shell=True)

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
                        break
                    if lines.split()[1].split(',')[1] == "R":
                        resname = lines.split()[2]
                        total_energy = float(lines.split(",")[-3])
                        if lines.split()[3] not in rec_decomp_results:
                            rec_decomp_results[resname] = total_energy
                        else:
                            rec_decomp_results[resname] += total_energy
                    if lines.split()[1].split(',')[1] == "L":
                        resname = lines.split()[2]
                        total_energy = float(lines.split(",")[-3])
                        if lines.split()[3] not in lig_decomp_results:
                            lig_decomp_results[resname] = total_energy
                        else:
                            lig_decomp_results[resname] += total_energy

        CleanDecompHis()
        WriteTotalFromDecomp()
        GetResults()
        return rec_decomp_results, lig_decomp_results

    def ParallelFeaturize(self) -> None:
        cpuUnits = int(cpu_count() // 2)
        manager = Manager()
        shared_list = manager.list()  # Create a shared list
        with Pool(processes=cpuUnits) as p:
            for pdbID, chains in self.dbDict.items():
                p.apply_async(self.Featurize, args=(pdbID, chains, shared_list,))
            p.close()
            p.join()
        for df in shared_list:
            self.dataframe = pd.concat([self.dataframe, df])

    def GetDataAsDataframe(self) -> pd.DataFrame:
        return self.dataframe
