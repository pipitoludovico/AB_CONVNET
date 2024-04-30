import os

import pandas as pd
from subprocess import Popen

from multiprocessing import Pool, cpu_count


def MoveToSelected(m_chunk: pd.DataFrame, dbFolder):
    m_chunk.apply(lambda x: Popen(f"cp {dbFolder}/{x.pdb}.pdb ./selected", shell=True).wait(), axis=1)


class DatabaseManager:
    def __init__(self, database, dbFolder):
        self.database = database
        self.dbFolder = dbFolder
        self.processes = []
        self.complex_data = {}
        self.quarterCPUcount = int((cpu_count()) / 4)
        os.makedirs('./selected', exist_ok=True)

    def CopyFilesFromFolderToTarget(self, copy_=False):
        with Pool(processes=self.quarterCPUcount) as p:
            for chunk in pd.read_csv(f'{self.database}', header=0, chunksize=1000):
                for idx, row in chunk.iterrows():
                    self.complex_data[row.pdb] = [row.Hchain, row.Lchain, row.antigen_chain, row.antigen_type]
                if copy_:
                    self.processes.append(p.apply_async(MoveToSelected, args=(chunk, self.dbFolder)))
            for _ in self.processes:
                _.get()
            p.close()
            p.join()

        return self.complex_data
