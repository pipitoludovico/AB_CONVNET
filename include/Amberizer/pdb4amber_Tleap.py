import os.path
from subprocess import run, DEVNULL, CalledProcessError


def pdb4amber():
    os.makedirs('pdb4amber', exist_ok=True)
    _ = ['receptor', 'ligand', 'complex']
    for x in _:
        try:
            run(f'pdb4amber -i initial/{x}_initial.pdb -o pdb4amber/{x}.pdb -y -a --reduce >> logs/pdb4amber.log 2>&1',
                check=True,
                shell=True, stdout=DEVNULL)
        except CalledProcessError:
            raise IOError("pdb4amber failed generating the pdb file for PDB: ", os.getcwd())


def RunTleap() -> None:
    run('tleap -f inleap; mv leap.log logs', check=True, shell=True, stdout=DEVNULL)


def UpdateSSbonds():
    SSbonds = {}
    for file in os.listdir("pdb4amber"):
        if "_sslink" in file and os.path.getsize("pdb4amber/" + file) != 0:
            partName = str(file[0:3])
            SSbonds[partName] = []
            with open("pdb4amber/" + file, 'r') as ssFile:
                for line in ssFile.readlines():
                    SSbonds[partName].append((line.split()[0], line.split()[1]))
    with open('inleap_mod', 'w') as moddedInleap:
        with open('inleap', 'r') as modInleap:
            for line in modInleap.readlines():
                moddedInleap.write(line)
                if 'rec = loadpdb' in line:
                    for key, value in SSbonds.items():
                        if key == 'rec':
                            for listOfvalues in SSbonds[key]:
                                moddedInleap.write(f"bond {key}.{listOfvalues[0]}.SG {key}.{listOfvalues[1]}.SG\n")
                if 'lig = loadpdb' in line:
                    for key, value in SSbonds.items():
                        if key == 'lig':
                            for listOfvalues in SSbonds[key]:
                                moddedInleap.write(f"bond {key}.{listOfvalues[0]}.SG {key}.{listOfvalues[1]}.SG\n")
                if 'com = loadpdb' in line:
                    for key, value in SSbonds.items():
                        if key == 'com':
                            for listOfvalues in SSbonds[key]:
                                moddedInleap.write(f"bond {key}.{listOfvalues[0]}.SG {key}.{listOfvalues[1]}.SG\n")
