import os


def WritePBSAinput():
    pbsa = ('&general\n', '\tkeep_files=0, start=1,\n', '/\n', '&gb\n',
            '\tigb=8, saltcon=0.150,\n', '/\n',
            '&decomp\n', '\tidecomp=1, dec_verbose=1, print_res="all"\n', '/\n')

    with open('mmgbsa.in', 'w') as mmgbsa:
        for line_ in pbsa:
            mmgbsa.write(line_)


def WriteTleapInput():
    os.makedirs('gbsa', exist_ok=True)
    _ = ["source leaprc.protein.ff19SB", "source leaprc.gaff2", "source leaprc.water.tip3p",
         "set default PBRadii mbondi3",
         'rec = loadpdb "initial/receptor_initial.pdb"', "check rec", 'saveamberparm rec gbsa/receptor.prmtop gbsa/receptor.inpcrd',
         'lig = loadpdb "initial/ligand_initial.pdb"', 'check lig', 'saveamberparm lig gbsa/ligand.prmtop gbsa/ligand.inpcrd',
         'com = loadpdb "initial/complex_initial.pdb"', 'check com', 'saveamberparm com gbsa/complex.prmtop gbsa/complex.inpcrd',
         'quit']

    with open('inleap', 'w') as inleap:
        for line in _:
            inleap.write(line + "\n")
