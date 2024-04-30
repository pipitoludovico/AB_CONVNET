from os import path, getcwd
import pandas as pd
import sys
from Amberizer.DbManager import DatabaseManager
from Amberizer.TrajectoryMaker import TrajectoryMaker
from Interfacer.PeStO import *
from Featurizer.FeatureMaker import FeaturizerClass
from warnings import filterwarnings

filterwarnings(action='ignore')

root = getcwd()


def main():
    if len(sys.argv) < 2:
        print("argv1= db.csv argv2=folder with pdbs")
        exit()
    csvDb = path.abspath(sys.argv[1])
    pdbsFolder = path.abspath(sys.argv[2])
    dbManager = DatabaseManager(csvDb, pdbsFolder)
    dbDict = dbManager.CopyFilesFromFolderToTarget(copy_=True)
    TrajMaker = TrajectoryMaker(dbDict)
    TrajMaker.ParallelPipeline()
    ParallelPesto(dbDict, root)
    featurizer = FeaturizerClass(dbDict, root)
    featurizer.ParallelFeaturize()
    df = featurizer.GetDataAsDataframe()
    pd.DataFrame.to_csv(df, 'zio.csv', index=True)
    nan_count = df.isna().sum()
    print(nan_count)


if __name__ == '__main__':
    main()
