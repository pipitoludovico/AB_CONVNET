import os.path
from os import chdir, cpu_count
from subprocess import run
from multiprocessing import Pool


def RunPesto(pdb, root) -> None:
    if not os.path.exists('selected/' + pdb + "/minimal_interface/selection_i0.pdb"):
        chdir("selected/" + pdb)
        run(f"python /home/scratch/software/ludovico/PeSTo/apply_model.py minimal_interface > logs/pesto.log 2>&1", shell=True)
    chdir(root)


def ParallelPesto(dbDict: dict, root) -> None:
    cpuUnits = int(cpu_count() // 2)
    processes = []
    with Pool(processes=cpuUnits) as p:
        for pdb, chains in dbDict.items():
            processes.append(p.apply_async(RunPesto, args=(pdb, root,)))
        for proc in processes:
            proc.get()
        p.close()
        p.join()
