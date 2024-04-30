import os
from os import chdir
from subprocess import run
import MDAnalysis as Mda
import numpy as np


def CopyAndSplitSystem(ROOT: str, pdb: str, chains: list, htmd: bool, x_range, y_range, z_range) -> None:
    if not htmd:
        u = Mda.Universe(f'{pdb}.pdb')
        u.dimensions = np.array([[x_range, y_range, z_range, 90, 90, 90]])
        SeparateComplex(u, chains)
    else:
        try:
            run(f"prepareProtein.py {pdb}.pdb cytosol", shell=True)
            u = Mda.Universe(f'protein_H.pdb')
            SeparateComplex(u, chains)
        except Exception as e:
            print(repr(e))
            chdir(ROOT)
            with open('failed.txt', 'a') as failFile:
                failFile.write(pdb + " failed.\n")
            return


def SeparateComplex(universe, chains: list):
    os.makedirs('initial', exist_ok=True)
    if not os.path.exists("initial/complex_initial.pdb"):
        sel = universe.select_atoms(f'protein or nucleic and chainID {chains[0]} {chains[1]} and not name H*')
        sel.write('initial/receptor_initial.pdb')
        sel = universe.select_atoms(f'protein or nucleic and chainID {chains[2]} and not name H*')
        sel.write('initial/ligand_initial.pdb')
        sel = universe.select_atoms(
            f'protein or nucleic and (chainID {chains[0]} {chains[1]} {chains[2]} and not name H*)')
        sel.write('initial/complex_initial.pdb')
