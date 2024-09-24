import warnings
import argparse
from os import path, getpid

warnings.filterwarnings(action='ignore')


def dir_path(string):
    if path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


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
    ap.add_argument('-database', type=dir_path)
    ap.add_argument('-train', required=False, action='store_true', help="Train the model with the saved database.")
    ap.add_argument('-test', required=False, action='store_true', help="Test the model with the sample/s")
    ap.add_argument("-k", '--kill', required=False, action='store_true', default=False,
                    help="Kills the PID in case of need.")
    args = ap.parse_args()
    with open('.mypid', 'w') as pidFile:
        pid = getpid()
        pidFile.write(str(pid))
    if args.kill is True:
        import os
        os.system('val=$(<.mypid ) && kill -15 -$val')
        exit()
    return vars(args)
