from subprocess import run
# from openmm.app import *
# from openmm.unit import *
# from openmm import *
#
#
# def LocalMinimization():
#     """Deprecated due to inconsistencies between pdb4amber and the Modeller builder pdbFixer"""
#     pdb = "pdb4amber/complex.pdb"
#     pdb = PDBFile(pdb)
#     modeller = Modeller(pdb.topology, pdb.positions)
#     forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
#     modeller.addHydrogens(forcefield)
#     system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=1 * nanometer,
#                                      constraints=HBonds)
#     integrator = LangevinMiddleIntegrator(300 * kelvin, 1 / picosecond, 0.004 * picoseconds)
#     simulation = Simulation(modeller.topology, system, integrator)
#     simulation.context.setPositions(modeller.positions)
#     simulation.minimizeEnergy(tolerance=0.1 * kilojoule / mole)
#
#     positions = simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions()
#     PDBFile.writeFile(simulation.topology, positions, open('complex_minimized.pdb', 'w'), keepIds=True)


def AMBERLocalMinimization():
    sanderInputMinimization = [
        "Minimization of the system",
        " &cntrl",
        "  imin=1,       ! Minimization",
        "  maxcyc=2000,  ! Maximum number of cycles",
        "  ncyc=500,     ! Print frequency",
        "  ntpr=100,     ! Print frequency",
        "  ntwx=100,     ! Write trajectory frequency",
        "  cut=12.0,     ! Cutoff for nonbonded interactions",
        "  ntb=1,        ! Periodic boundary conditions (constant volume)",
        "  igb=0,        ! No implicit solvent",
        " /"
    ]

    with open('minimize.in', 'w') as sandFile:
        for line in sanderInputMinimization:
            sandFile.write(line + "\n")

    run('sander -O -i minimize.in -o mdout -p gbsa/complex.prmtop -c gbsa/complex.inpcrd -r minimized_complex.rstr',
        check=True,
        shell=True)
    run("ambpdb -p gbsa/complex.prmtop -c minimized_complex.rstr > complex_minimized.pdb", check=True, shell=True)
