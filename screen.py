import os
from dataset import mol_data
from model import DTI_predictor
from predictor import Predictor
from preprocess.preprocessing import process_pdb
import time

dude_path = 'dataset/dude/all/'
target_names = os.listdir(dude_path)

target_names = ['abl1']
start = time.time()
for n, i in enumerate(target_names):
    print('{}/{} predicting {}.'.format(n+1, len(target_names), i))
    root = os.path.join(dude_path, i)
    pro_data = process_pdb(pocket='{}/{}_pocket_6A.pdb'.format(root, i),
                           pdb='{}/{}_fix.pdb'.format(root, i))
    model = DTI_predictor(ckpt_file='checkpoint/348_pearson_model.ckpt', pro_data=pro_data)

    for j in ['decoys_final.ism', 'actives_final.ism']:
        smiles_list = open(os.path.join(root, j), 'r').readlines()
        smiles_list = [_.split()[0] for _ in smiles_list]
        data = mol_data(root=root, smiles_list=smiles_list)
        predictor = Predictor(model, data, batch_size=512, num_workers=8)
        result = predictor.test()
        print(len(smiles_list))

end = time.time()
print(end-start)
