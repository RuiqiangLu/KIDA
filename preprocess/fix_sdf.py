import os
from openbabel import openbabel
from rdkit.Chem import SDWriter, SDMolSupplier
import dpdata


obConversion = openbabel.OBConversion()
obConversion.SetInAndOutFormats('sdf', 'sdf')

datadir = '../dataset/pdbbind/total-set/'
pdbids = os.listdir(datadir)
error_file = []

for pdbid in pdbids:
    mol = openbabel.OBMol()
    infile = os.path.join(datadir, pdbid, "{}_ligand.sdf".format(pdbid))
    obConversion.ReadFile(mol, infile)
    mol.AddHydrogens()
    addH = os.path.join(datadir, pdbid, "{}_addH.sdf".format(pdbid))
    obConversion.WriteFile(mol, addH)
    fixed = os.path.join(datadir, pdbid, "{}_fixed.sdf".format(pdbid))
    try:
        bondFixer = dpdata.BondOrderSystem(addH, sanitize_level='high', verbose=False, raise_errors=False)
        mol = bondFixer.rdkit_mol
        writer = SDWriter(fixed)
        writer.write(mol)
        writer.flush()
    except TypeError:
        print('Can not fix {}'.format(pdbid))
    os.remove(addH)

    # test
    if os.path.exists(fixed):
        mol = SDMolSupplier(fixed)[0]
        if mol is None:
            error_file.append(pdbid)
    else:
        error_file.append(pdbid)

print(error_file)
print('There are {}/{} cannot fix'.format(len(error_file), len(pdbids)))

# conda install -c conda-forge rdkit=2022.03.5 openbabel=3.1.1 dpdata=0.2.8
