import os.path

import time
from multiprocessing import Pool, cpu_count
from os import makedirs

from .pdb4amber_Tleap import *
from .ComplexMaker import *
from .Minimizer import *
from Interfacer.VMD import GetCloseSelection

begin = time.perf_counter()


def RunMMPBSA():
    run("$AMBERHOME/bin/MMPBSA.py -i mmgbsa.in -o results_mmgbsa.dat -cp gbsa/complex.prmtop -rp gbsa/receptor.prmtop -lp gbsa/ligand.prmtop -y gbsa/complex.dcd -eo gbsa.csv > logs/GBSA.log 2>&1",
        check=True, shell=True, stdout=DEVNULL)


def GetCrystalCoords(filePath):
    with open(f'{filePath}', 'r') as f:
        lines = f.readlines()
    x_coords, y_coords, z_coords = [], [], []
    for line in lines:
        if line.startswith('ATOM'):
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            x_coords.append(x)
            y_coords.append(y)
            z_coords.append(z)

    x_range = np.max(x_coords) - np.min(x_coords)
    y_range = np.max(y_coords) - np.min(y_coords)
    z_range = np.max(z_coords) - np.min(z_coords)
    return x_range, y_range, z_range


class TrajectoryMaker:
    def __init__(self, dbDict):
        self.dbDict = dbDict
        self.ROOT = os.getcwd()

    @staticmethod
    def TrajectorizePDB(chains) -> None:
        if not os.path.exists('complex_minimized.pdb'):
            # LocalMinimization()
            AMBERLocalMinimization()
        if os.path.exists('mdcrd') and os.path.exists('mdout'):
            run("rm mdcrd mdout", shell=True)
        GetCloseSelection((chains[0], chains[1]), chains[2])
        u = Mda.Universe('gbsa/complex.prmtop', 'complex_minimized.pdb')
        x_range, y_range, z_range = GetCrystalCoords(f"complex_minimized.pdb")
        u.dimensions = np.array([[x_range, y_range, z_range, 90, 90, 90]])
        sel = u.select_atoms(f'all')
        sel.write('gbsa/complex.dcd')

    def MakeTrajectoryFromPDB(self, pdb, chains) -> None:
        chdir('./selected')
        makedirs(f'{pdb}', exist_ok=True)
        run(f'mv {pdb}.pdb {pdb}/{pdb}.pdb', shell=True)
        chdir(pdb)
        os.makedirs('./logs', exist_ok=True)

        def DoStuff(useHTMD):
            x_range, y_range, z_range = GetCrystalCoords(f"{pdb}.pdb")
            if not os.path.exists('initial') or not os.path.exists('gbsa') or not os.path.exists('pdb4amber'):
                CopyAndSplitSystem(self.ROOT, pdb, chains, useHTMD, x_range, y_range, z_range)
                RunTleapANDpdb4amber()

        def CheckIfOK():
            if not os.path.exists('gbsa/complex.prmtop') or not os.path.exists('gbsa/complex.inpcrd'):
                try:
                    DoStuff(useHTMD=True)
                    CheckIfOK()
                except:
                    os.chdir(self.ROOT)
                    with open('failed.txt', 'a') as failFile:
                        failFile.write(pdb + " failed.\n")
                    raise IOError('Featurizer failed in', os.getcwd())

        if not os.path.exists('gbsa/') and not os.path.exists('initial/'):
            DoStuff(useHTMD=False)
            CheckIfOK()
        if not os.path.exists('gbsa/complex.dcd'):
            self.TrajectorizePDB(chains)
            WritePBSAinput()
        if not os.path.exists('results_mmgbsa.dat') and not os.path.exists('FINAL_DECOMP_MMPBSA.dat'):
            RunMMPBSA()
        chdir(self.ROOT)

    def ParallelPipeline(self) -> None:
        cpuUnits = int(cpu_count() // 2)
        processes = []
        with Pool(processes=cpuUnits) as p:
            for pdb, chains in self.dbDict.items():
                processes.append(p.apply_async(self.MakeTrajectoryFromPDB, args=(pdb, chains,)))
            for proc in processes:
                proc.get()
            p.close()
            p.join()
