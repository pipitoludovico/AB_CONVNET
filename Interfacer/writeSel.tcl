package require psfgen
resetpsf
coordpdb ['H', 'L', 'N']
set sel [atomselect top "(not chain H and same residue as protein within 10 of chain H) or (not chain L and same residue as protein within 10 of chain L)"]
$sel writepdb selection.pdb
quit
