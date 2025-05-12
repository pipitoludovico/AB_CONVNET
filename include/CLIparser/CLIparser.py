import warnings
import argparse
from os import path, getpid

warnings.filterwarnings(action='ignore')


def dir_path(string):
    if path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def check_existence(string):
    if path.exists(string):
        return string
    else:
        raise FileNotFoundError(string)


def file_path(string):
    if path.isfile(string):
        return string
    else:
        raise NotADirectoryError(string)


def ParseCLI():
    ap = argparse.ArgumentParser()
    ap.add_argument('-dp', '--datapipeline', required=False, action='store_true',
                    help="Convert the database to matrices.")
    ap.add_argument('-csv', type=file_path)
    ap.add_argument('-database', type=dir_path, help="The folder with the pdb files.")
    ap.add_argument('-train', required=False, action='store_true', help="Train the model with the saved database.")
    ap.add_argument('-samplesToTest', required=False, type=check_existence, help="Specify a folder with PDBs to use for testing the NN.")
    ap.add_argument('-test', required=False, action='store_true', help="Test the model with the saved database.")
    ap.add_argument('-test2', required=False, action='store_true', help="Test the model on the training database.")
    ap.add_argument('-lr', required=False, type=float, default=1e-4,
                    help="sets the learning rate for Adam optimized [default = 1e-5]")
    ap.add_argument('-l2', required=False, type=float, default=1e-2,
                    help="sets the learning rate for bias regulizer [default = 1e-2]")
    ap.add_argument('-batch', required=False, type=int, default=32, help="sets the batch size [default = 32]")
    ap.add_argument('-epoch', required=False, type=int, default=50, help="sets the number of epochs [default = 50]")
    ap.add_argument('-split', required=False, type=int, default=80, help="sets % split between test and train [default = 80]")

    ap.add_argument('-gan', required=False, action='store_true', help="Tries to train the GAN")

    ap.add_argument('-format', required=False, action='store_true', help="format data and removes quartiles 15-85")
    ap.add_argument('-model', required=False, type=str, help="specify the file path with the keras model")
    ap.add_argument('-name', required=False, type=str, default='default', help="names the model's file name")
    ap.add_argument("-k", '--kill', required=False, action='store_true', default=False,
                    help="Kills the PID in case of need.")
    args = ap.parse_args()
    if args.kill is True:
        import os
        os.system('val=$(<.mypid ) && kill -15 -$val')
        exit()

    with open('.mypid', 'w') as pidFile:
        pid = getpid()
        pidFile.write(str(pid))

    if args.datapipeline:
        if not args.csv or not args.database:
            ap.error("When using --datapipeline, both --csv and --database are required.")
    return vars(args)
