import os
from os import path, getcwd
import sys
from Amberizer.DbManager import DatabaseManager
from Amberizer.TrajectoryMaker import TrajectoryMaker
from Interfacer.PeStO import *
from Featurizer.FeatureMaker import FeaturizerClass
from Model.MLR import LoadData

from warnings import filterwarnings
from CLIparser.CLIparser import ParseCLI

filterwarnings(action='ignore')

root = getcwd()


def main():
    args = ParseCLI()
    if args['datapipeline']:
        csvDb = path.abspath(args['csv'])
        pdbsFolder = path.abspath(args['database'])
        dbManager = DatabaseManager(csvDb, pdbsFolder)

        dbDict = dbManager.CopyFilesFromFolderToTarget(copy_=True)
        TrajMaker = TrajectoryMaker(dbDict)
        TrajMaker.ParallelPipeline()
        ParallelPesto(dbDict, root)
        featurizer = FeaturizerClass(dbDict, root)
        featurizer.ParallelFeaturize()
    if args['train']:
        LoadData()


if __name__ == '__main__':
    main()
