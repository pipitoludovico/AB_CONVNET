from include.Amberizer.DbManager import DatabaseManager
from include.Amberizer.TrajectoryMaker import TrajectoryMaker
from include.Featurizer.FeatureMaker import ParallelFeaturize
from include.Interfacer.PeStO import ParallelPesto
from Model.MatrixFormatter import FormatData
from Model.DiscriminatorTraining import Train
from Model.Sampler import Sampler
from Model.Test import *
from Model.cGAN_Training import TrainAndGenerate
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
        ParallelFeaturize(dbDict, root)
    if args['format']:
        FormatData()
    if args['train']:
        Train(args)
    if args['test']:
        path_ = args.get('samplesToTest') if args.get('samplesToTest') is not None else './test'
        print("Samples location path:", path_)
        Sampler(path_)
        Test(args=args)
    if args['test2']:
        Test2(args=args)
    if args['gan']:
        TrainAndGenerate(pretrained_discriminator_model_path=args['model'])


if __name__ == '__main__':
    main()
