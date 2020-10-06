from sys import argv

from rdkit.Chem import SDMolSupplier, SmilesWriter

sdm = SDMolSupplier(argv[1])
sw = SmilesWriter(argv[2], includeHeader=False, nameHeader='_Name')
for mol in sdm:
    sw.write(mol)
sw.close()
