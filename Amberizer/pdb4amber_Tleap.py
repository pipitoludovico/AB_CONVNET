from subprocess import run, DEVNULL, CalledProcessError
from .InputWriter import *


def pdb4amber(addAt=False):
    os.makedirs('pdb4amber', exist_ok=True)
    if addAt:
        addA = '--add-missing-atoms'
    else:
        addA = ''
    _ = ['receptor', 'ligand', 'complex']
    for x in _:
        try:
            run(f'pdb4amber -i initial/{x}_initial.pdb -o pdb4amber/{x}.pdb -y -a {addA} > logs/pdb4amber.log 2>&1',
                check=True,
                shell=True, stdout=DEVNULL)
        except CalledProcessError:
            raise IOError("pdb4amber failed generating the pdb file for PDB: ", os.getcwd())


def RunTleapANDpdb4amber() -> None:
    addMissingAtoms = False
    WriteTleapInput()

    try:
        pdb4amber(addMissingAtoms)
        run('tleap -f inleap; mv leap.log logs', check=True, shell=True, stdout=DEVNULL)
    except CalledProcessError:
        addMissingAtoms = True
        pdb4amber(addMissingAtoms)
        run('tleap -f inleap;mv leap.log logs/;echo "ADDING ATOMS VIA PDB4AMBER ATTEMP" >> logs/leap.log', check=True, shell=True, stdout=DEVNULL)
