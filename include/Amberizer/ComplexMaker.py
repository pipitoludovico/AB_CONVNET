import MDAnalysis as Mda
import warnings
import numpy as np

from .InputWriter import *
from .pdb4amber_Tleap import *

warnings.filterwarnings(action='ignore')


def CopyAndSplitSystem(pdb: str, chains: list, x_range, y_range, z_range) -> None:
    try:
        u = Mda.Universe(pdb)
        u.dimensions = np.array([[x_range, y_range, z_range, 90, 90, 90]])
        SeparateComplex(u, chains)
    except Exception as e:
        print(repr(e))
        with open('../../failed.txt', 'a') as failFile:
            failFile.write(pdb + " failed.\n")
        return


def SeparateComplex(universe, chains: list):
    os.makedirs('initial', exist_ok=True)
    if not os.path.exists("initial/complex_initial.pdb"):
        if len(chains) > 2:
            sel = universe.select_atoms(f'protein or nucleic and chainID {chains[0]} {chains[1]}')
            sel.write('initial/receptor_initial.pdb')
            sel = universe.select_atoms(f'protein or nucleic and chainID {chains[2]}')
            sel.write('initial/ligand_initial.pdb')
            sel = universe.select_atoms(
                f'protein or nucleic and (chainID {chains[0]} {chains[1]} {chains[2]})')
            sel.write('initial/complex_initial.pdb')
        if len(chains) == 2:
            sel = universe.select_atoms(f'protein or nucleic and chainID {chains[0]}')
            sel.write('initial/receptor_initial.pdb')
            sel = universe.select_atoms(f'protein or nucleic and chainID {chains[1]}')
            sel.write('initial/ligand_initial.pdb')
            sel = universe.select_atoms(f'protein or nucleic and (chainID {chains[0]} {chains[1]})')
            sel.write('initial/complex_initial.pdb')


def RestoreChains(pdb):
    originalChains_: list = []
    with open(f"{pdb}_pdb4amber.pdb", 'r') as pdb4amberFile:
        for pdb4amberLine in pdb4amberFile.readlines():
            if "ATOM" in pdb4amberLine:
                if pdb4amberLine[21] not in originalChains_:
                    originalChains_.append(pdb4amberLine[21])
    count = 0
    chainsLeft = []
    with open("complex_minimized.pdb", 'r') as minimizedPDB:
        minimizedLines = minimizedPDB.readlines()
        with open('complex_minimized_chains.pdb', 'w') as test:
            for line in minimizedLines:
                if len(line.split()) > 2:
                    newline = line[:21] + originalChains_[count] + line[22:]
                    if "TER" in line:
                        newline = line
                        count += 1
                    test.write(newline)
                else:
                    test.write(line)

        with open(f"complex_minimized_chains.pdb", 'r') as minimized_chains:
            for min_chain_line in minimized_chains.readlines():
                if "ATOM" in min_chain_line:
                    if min_chain_line[21] not in chainsLeft:
                        chainsLeft.append(min_chain_line[21])

        return chainsLeft


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


def SplitAndTleap(pdb, chains):
    x_range, y_range, z_range = GetCrystalCoords(pdb)
    if not os.path.exists('initial') or not os.path.exists('gbsa') or not os.path.exists('pdb4amber'):
        CopyAndSplitSystem(pdb, chains, x_range, y_range, z_range)
        WriteTleapInput()
        RunTleap()
