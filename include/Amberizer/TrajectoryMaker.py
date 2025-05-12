import os.path
from os import chdir
from multiprocessing import Pool, cpu_count

from .ComplexMaker import *
from .Minimizer import *
from include.Interfacer.VMD import GetVDWcontacts


def RunMMPBSA() -> None:
    run(f"MMPBSA.py -i mmgbsa.in -o results_mmgbsa.dat -cp gbsa/complex.prmtop -rp gbsa/receptor.prmtop -lp gbsa/ligand.prmtop -y complex_minimized_chains.pdb -eo gbsa.csv > logs/GBSA.log 2>&1",
        shell=True, stdout=DEVNULL, stderr=DEVNULL)
    if not os.path.exists('results_mmgbsa.dat'):
        run(f"touch GBSA_FAILED", shell=True)


def TrajectorizePDB() -> None:
    if not os.path.exists('mdcrd') and not os.path.exists('mdout'):
        AMBERLocalMinimization()


def GetChains(pdb_):
    if os.path.exists('mdcrd') and os.path.exists('mdout') and os.path.exists('complex_minimized.pdb'):
        with open('complex_minimized.pdb', 'r') as minimizedPDB:
            fileContent = minimizedPDB.read()
            if "nan" not in fileContent or "*" not in fileContent:
                chainIDs_ = RestoreChains(pdb_)
                return chainIDs_
            else:
                run("ambpdb -p gbsa/complex.prmtop -c gbsa/complex.inpcrd > complex_minimized.pdb",
                    shell=True)  # temp has no chain info
                chainIDs_ = RestoreChains(pdb_)
                return chainIDs_
    return None


class TrajectoryMaker:
    def __init__(self, dbDict):
        self.dbDict = dbDict
        self.ROOT = os.getcwd()

    def MakeTrajectoryFromPDB(self, pdb: str, chains_) -> None:
        chdir(f'./selected/{pdb}')
        os.makedirs('./logs', exist_ok=True)
        if not os.path.exists('fail'):
            try:
                if not os.path.exists(f"{pdb}_pdb4amber.pdb"):
                    run(f"pdb4amber -i {pdb}.pdb -o {pdb}_pdb4amber.pdb -y -d -a -p >> logs/initial_pdb4amber.log 2>&1;rm {pdb}_pdb4amber_*;",
                        shell=True)  # cleaning the structure
                if not os.path.exists('gbsa/') or not os.path.exists('initial/'):
                    SplitAndTleap(pdb=pdb + "_pdb4amber.pdb", chains=chains_)
                if not os.path.exists('complex_minimized.pdb'):
                    TrajectorizePDB()  # this makes complex_minimized_chains.pdb
                if os.path.exists('complex_minimized_chains.pdb'):
                    chainsID = GetChains(pdb)
                    if not os.path.exists('contacts.int'):
                        if chainsID:
                            if len(chainsID) >= 3:
                                try:
                                    GetVDWcontacts((chainsID[0], chainsID[1]), chainsID[2])
                                except:
                                    GetVDWcontacts((chainsID[0], chainsID[0]), chainsID[1])
                            if len(chainsID) == 2:
                                try:
                                    GetVDWcontacts((chainsID[0], chainsID[0]), chainsID[1])
                                except:
                                    GetVDWcontacts((chainsID[0], chainsID[1]), chainsID[0])
                if not os.path.exists('results_mmgbsa.dat') and not os.path.exists(
                        'FINAL_DECOMP_MMPBSA.dat') and not os.path.exists("GBSA_FAILED"):
                    WritePBSAinput()
                    RunMMPBSA()
            except:
                run('touch fail', shell=True)
                with open('../../failed.txt', 'a') as failFile:
                    failFile.write(pdb + "  GBSA failed.\n")
                chdir(self.ROOT)
        chdir(self.ROOT)

    def ParallelPipeline(self) -> None:
        cpuUnits = int(cpu_count() // 4)
        processes = []
        with Pool(processes=cpuUnits) as p:
            for pdb, chains in self.dbDict.items():
                processes.append(p.apply_async(self.MakeTrajectoryFromPDB, args=(pdb, chains,)))
            for proc in processes:
                proc.get()
            p.close()
            p.join()
