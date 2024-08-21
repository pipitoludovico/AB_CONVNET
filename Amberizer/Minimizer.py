import os.path
from subprocess import run


def AMBERLocalMinimization():
    sanderInputMinimization = [
        "Minimization of the system", " &cntrl", "  imin=1,       ! Minimization",
        "  maxcyc=1000,  ! Maximum number of cycles", "  ntpr=100,     ! Print frequency",
        "  ntwx=500,     ! Write trajectory frequency", "  cut=12.0,     ! Cutoff for nonbonded interactions",
        "  ntb=0,        ! Periodic boundary conditions (constant volume)", "  igb=8,        ! No implicit solvent",
        " /"]
    with open('minimize.in', 'w') as sandFile:
        for line in sanderInputMinimization:
            sandFile.write(line + "\n")

    with open('sanderOut.out', 'w') as sanderOut, open('sanderErr.err', 'w') as sanderErr:
        run('sander -O -i minimize.in -o mdout -p gbsa/complex.prmtop -c gbsa/complex.inpcrd -r minimized_complex.rstr',
            shell=True, stdout=sanderOut, stderr=sanderErr)
        if os.path.exists('gbsa/complex.prmtop') and os.path.exists('minimized_complex.rstr'):
            run("ambpdb -p gbsa/complex.prmtop -c minimized_complex.rstr > complex_minimized.pdb", shell=True,
                stdout=sanderOut, stderr=sanderErr)
    if os.path.getsize('sanderOut.out') == 0:
        run('rm sanderOut.out', shell=True)
    if os.path.getsize("sanderErr.err") == 0:
        run('rm sanderErr.err', shell=True)
