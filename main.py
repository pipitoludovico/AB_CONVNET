from include.Amberizer.DbManager import DatabaseManager
from include.Amberizer.TrajectoryMaker import TrajectoryMaker
from include.Featurizer.FeatureMaker import FeaturizerClass
from include.Interfacer.PeStO import ParallelPesto
from Model.MatrixFormatter import *
from Model.BuildMLR import *
from Model.FeatureExtractor import *
from Model.TestModel import *
from warnings import filterwarnings
from include.CLIparser.CLIparser import ParseCLI

filterwarnings(action='ignore')

root = getcwd()


def main():
    args = ParseCLI()
    print(args)
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
        FormatData()
        TrainModel(args)
    if args['test']:
        path_ = args['test']
        matrixData_ = GetFeatures(path_)
        Test(matrixData_)


if __name__ == '__main__':
    main()
