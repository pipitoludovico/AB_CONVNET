import os
from subprocess import run, DEVNULL


def RestoreChains():
    chains_: list = []
    with open("pdb4amber/complex.pdb", 'r') as pdb4amberFile:
        for pdb4amberLine in pdb4amberFile.readlines():
            if "ATOM" in pdb4amberLine:
                if pdb4amberLine[21] not in chains_:
                    chains_.append(pdb4amberLine[21])
    count = 0
    with open("complex_minimized.pdb", 'r') as minimizedPDB:
        minimizedLines = minimizedPDB.readlines()
        with open('temp.pdb', 'w') as test:
            for line in minimizedLines:
                if len(line.split()) > 2:
                    newline = line[:21] + chains_[count] + line[22:]
                    if "TER" in line:
                        newline = line
                        count += 1
                    test.write(newline)
                else:
                    test.write(line)


def GetVDWcontacts(abChains, agChains):
    tcl_script_lines = [
        'mol load pdb complex_minimized.pdb',
        '',
        'proc contactFreq {sel1 sel2 outFile mol} {',
        '  set allCounts {}',
        '  set numberOfFrames [molinfo $mol get numframes]',
        '',
        '  if { $outFile != "stdout" } {',
        '     set outFile [open $outFile w]',
        '  }',
        '',
        '  for {set i 0} {$i < $numberOfFrames} {incr i} {',
        '    molinfo $mol set frame $i',
        '',
        '    set frameCount1 [atomselect $mol "$sel1 and noh and within 3.5 of ($sel2 and noh)"]',
        '    set frameCount2 [atomselect $mol "$sel2 and noh and within 3.5 of ($sel1 and noh)"]',
        '',
        '    set uniqueContacts [list]',
        '',
        '    foreach a [$frameCount1 get {resname segid}] {',
        '      foreach b [$frameCount2 get {resname segid}] {',
        '        set contact [concat $a "-" $b]',
        '        if {[lsearch -exact $uniqueContacts $contact] == -1} {',
        '          lappend uniqueContacts $contact',
        '        }',
        '      }',
        '    }',
        '',
        '    set numContacts [llength $uniqueContacts]',
        '',
        '    lappend allCounts $numContacts',
        '',
        '    if { $outFile == "stdout" } {',
        '      puts "Frame $i $numContacts contacts"',
        '    } else {',
        '      set contactInfo {}',
        '      foreach contact $uniqueContacts {',
        '        lappend contactInfo [join $contact ""]',
        '      }',
        '      puts $outFile "$numContacts,[join $contactInfo ","]"',
        '    }',
        '',
        '    $frameCount1 delete',
        '    $frameCount2 delete',
        '  }',
        '',
        '  if { $outFile != "stdout" } {',
        '    close $outFile',
        '  }',
        '}',
        '',
        f'set sel1 "chain {" ".join(abChains)}"',
        f'set sel2 "chain {agChains}"',
        'set outName "contacts.int"',
        '',
        '',
        'puts "GETTING CONTACTS"',
        'contactFreq $sel1 $sel2 $outName top',
        '',
        'exit'
    ]

    with open('getContacts.tcl', 'w') as vmdContactsFile:
        for line in tcl_script_lines:
            vmdContactsFile.write(line + "\n")
    run(f'vmd -dispdev text -e getContacts.tcl > logs/getContacts.log', shell=True, check=True)


def GetCloseSelection(abChain, agChain):
    RestoreChains()
    os.makedirs('minimal_interface', exist_ok=True)
    selectVMD = ("package require psfgen",
                 "resetpsf",
                 f"mol load pdb temp.pdb",
                 f'set sel [atomselect top "(not chain {" ".join(abChain)} and same residue as protein within 10 of chain {" ".join(abChain)}) or'
                 f' (not chain {agChain} and same residue as protein within 10 of chain {agChain})"]',
                 "$sel writepdb minimal_interface/selection.pdb",
                 "quit")
    with open('writeSel.tcl', 'w') as writer:
        for line in selectVMD:
            writer.write(line + "\n")
    run('vmd -dispdev text -e writeSel.tcl > logs/writeSel.log 2>&1; mv temp.pdb complex_minimized.pdb', shell=True,
        stdout=DEVNULL)
