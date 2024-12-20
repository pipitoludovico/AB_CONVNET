from subprocess import run, DEVNULL


def GetVDWcontacts(filePath="complex_minimized_chains.pdb", abChains="H L", agChains="A",
                   outputPath="contacts.int"):
    tcl_script_lines = [f'mol load pdb {filePath}', '', 'proc contactFreq {sel1 sel2 outFile mol} {',
                        '  set allCounts {}', '  set numberOfFrames [molinfo $mol get numframes]',
                        '', '  if { $outFile != "stdout" } {', '     set outFile [open $outFile w]',
                        '  }', '', '  for {set i 0} {$i < $numberOfFrames} {incr i} {', '    molinfo $mol set frame $i',
                        '', '    set frameCount1 [atomselect $mol "$sel1 and noh and within 3.5 of ($sel2 and noh)"]',
                        '    set frameCount2 [atomselect $mol "$sel2 and noh and within 3.5 of ($sel1 and noh)"]',
                        '', '    set uniqueContacts [list]', '', '    foreach a [$frameCount1 get {resname resid}] {',
                        '      foreach b [$frameCount2 get {resname resid}] {',
                        '        set contact [concat [lindex $a 0][lindex $a 1] "-" [lindex $b 0][lindex $b 1]]',
                        '        if {[lsearch -exact $uniqueContacts $contact] == -1} {',
                        '          lappend uniqueContacts $contact', '        }', '      }', '    }', '',
                        '    set numContacts [llength $uniqueContacts]', '', '    lappend allCounts $numContacts', '',
                        '    if { $outFile == "stdout" } {', '      puts "Frame $i $numContacts contacts"',
                        '    } else {', '      set contactInfo {}', '      foreach contact $uniqueContacts {',
                        '        lappend contactInfo [join $contact ""]', '      }',
                        '      puts $outFile "$numContacts,[join $contactInfo ","]"', '    }', '',
                        '    $frameCount1 delete', '    $frameCount2 delete', '  }', '',
                        '  if { $outFile != "stdout" } {', '    close $outFile', '  }', '}', '',
                        f'set sel1 "protein and chain {" ".join(abChains).replace("|", " ")}"',
                        f'set sel2 "protein and chain {str(agChains).replace("|", " ")}"', f'set outName {outputPath}',
                        '', '', 'puts "GETTING CONTACTS"', 'contactFreq $sel1 $sel2 $outName top', '', 'exit']

    with open('getContacts.tcl', 'w') as vmdContactsFile:
        for line in tcl_script_lines:
            vmdContactsFile.write(line + "\n")
    run(f'vmd -dispdev text -e getContacts.tcl > logs/getContacts.log 2>&1', shell=True, check=True, stdout=DEVNULL)
