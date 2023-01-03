from pymol import cmd
import os

pdb_path = 'dataset/pdbbind/total-set/'
df = open('dataset/pdbbind/index/INDEX_general_PL_data.2016', 'r').readlines()[6:]

pdbbind_list = [i.split()[0] for i in df]
distance = 6
os.chdir(pdb_path)
for pdb in pdbbind_list:
    if not os.path.exists(pdb):
        continue
    os.chdir(pdb)
    protein = pdb + '_fix.pdb'
    state = '_fix '
    if not os.path.exists(protein):
        protein = pdb + '_protein.pdb'
        state = '_protein '
    ligand = pdb + '_ligand.sdf'
    cmd.load(protein)
    cmd.load(ligand)
    cmd.remove('solvent')
    cmd.create('complex', pdb + state + pdb + '_ligand')
    cmd.select('atoms', 'resn UNK around ' + str(distance))
    cmd.select('residues', 'byres atoms')
    cmd.save(pdb + '_pocket_' + str(distance) + 'A.pdb', 'residues')
    cmd.create('complex', 'residues ' + pdb + '_ligand')
    cmd.save(pdb + '_complex_' + str(distance) + 'A.pdb', 'complex')
    cmd.delete('all')
    os.chdir('..')
    print('extract', protein, 'pocket success')


###### pymol cmd #####
#  load protein_name.pdb
#  load ligand_name.sdf
#  remove solvent
#  create complex, protein_name ligand_name
#  select atoms, resn UNK around 6
#  select residues, byres atoms
#  create complex, residues ligand_name
#  save complex_name.pdb, complex
