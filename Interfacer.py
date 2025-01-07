from include.Amberizer.DbManager import DatabaseManager
from include.Amberizer.TrajectoryMaker import TrajectoryMaker
from include.Featurizer.FeatureMaker import FeaturizerClass
from include.Interfacer.PeStO import ParallelPesto
from Model.MatrixFormatter import *
from Model.Train import *
from Model.Sampler import *
from Model.Test import *
from warnings import filterwarnings
from include.CLIparser.CLIparser import ParseCLI

filterwarnings(action='ignore')

root = os.getcwd()


def main():
    args = ParseCLI()
    print(args)
    if args['datapipeline']:
        csvDb = path.abspath(args['csv'])
        pdbsFolder = path.abspath(args['database'])
        dbManager = DatabaseManager(csvDb, pdbsFolder)

        dbDict = dbManager.CopyFilesFromFolderToTarget(copy_=False)
        TrajMaker = TrajectoryMaker(dbDict)
        TrajMaker.ParallelPipeline()
        ParallelPesto(dbDict, root)
        featurizer = FeaturizerClass(dbDict, root)
        featurizer.ParallelFeaturize()
    if args['format']:
        FormatData()
    if args['train']:
        Train(args)
    if args['samplesToTest']:
        path_ = args['samplesToTest']
        GetFeatures(path_)
    if args['test']:
        Test(args=args)
    if args['test2']:
        Test2(args=args)


if __name__ == '__main__':
    main()
