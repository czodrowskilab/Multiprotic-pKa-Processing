"""
This script combines multiple SDF files specified via commandline arguments
to one SDF file. It will be written to stdout. The
script tries to preserve the information about the original dataset name by
splitting the file name at "_" and saving the first element of this split as
SDF file tag named "original_dataset".
"""

from os.path import basename
from sys import argv, stderr

from rdkit.Chem import SDWriter, SDMolSupplier

__author__ = 'Marcel Baltruschat'
__copyright__ = 'Copyright © 2020'
__license__ = 'MIT'
__version__ = '1.1.0'

sdw = SDWriter('-')
dropped = 0
for f in argv[1:]:
    dsname = basename(f)
    sdm = SDMolSupplier(f)
    for mol in sdm:
        if not mol:
            dropped += 1
            continue
        mol.SetProp('original_dataset', dsname)
        sdw.write(mol)
sdw.close()
print(f'Dropped {dropped} molecules due invalid molblocks', file=stderr)
