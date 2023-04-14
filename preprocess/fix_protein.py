from pdbfixer import PDBFixer
from simtk.openmm.app import PDBFile
# from openmm.app import PDBFile
from multiprocessing import Pool
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--file_path', type=str, default='../dataset/pdbbind/total-set/')
args = parser.parse_args()
df = open('../dataset/pdbbind/index/INDEX_general_PL_data.2016', 'r').readlines()[6:]
file = [i.split()[0] for i in df]


def process(i):
    filename = os.path.join(args.file_path, i, i + '_protein.pdb')
    if not os.path.exists(filename):
        return
    fixed_file = os.path.join(args.file_path, i, i + '_fix.pdb')
    fixer = PDBFixer(filename=filename)
    chains = list(fixer.topology.chains())
    fixer.findMissingResidues()
    keys = fixer.missingResidues.keys()
    missingResidues = dict()
    for key in keys:
        chain = chains[key[0]]
        if not(key[1] == 0 or key[1] == len(list(chain.residues()))):
            missingResidues[key] = fixer.missingResidues[key]
    fixer.missingResidues = missingResidues
    fixer.removeHeterogens(False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    PDBFile.writeFile(fixer.topology, fixer.positions, open(fixed_file, 'w'))
    # print('fixed {}'.format(i))
    return 0


pool = Pool(processes=14)
for i in file:
    pool.apply_async(process, args=(i, ))

pool.close()
pool.join()
