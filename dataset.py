import torch
import os
import numpy as np
import random
from torch_geometric.data import Data, Dataset, InMemoryDataset
from rdkit import Chem
from preprocess.preprocessing import process_mol


class PDBBind_Dataset(Dataset):
    def __init__(self, root, dataset='general', split='general-core',
                 transform=None, pre_transform=None, pre_filter=None):
        self.dataset = dataset
        self.dict_files = os.path.join(root, 'idx_dict.npy')
        self.file_names = []
        super(PDBBind_Dataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.train_idx, self.val_idx, self.test_idx = self.split(split)

    @property
    def raw_file_names(self):
        return

    @property
    def processed_file_names(self):
        file_names = []
        if os.path.exists(self.dict_files):
            idx_dict = np.load(self.dict_files, allow_pickle=True).item()
            idxs = list(idx_dict.values())
            for v in idxs:
                file_names.append('{}.pt'.format(v))
            return file_names
        else:
            self.process()
            return None

    def len(self):
        return int(self.processed_file_names[-1].split('.')[0]) + 1

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, '{}.pt'.format(idx)))
        return data

    def process(self):
        print('No data, prepared first!')
        return

    def split(self, split='general-core'):
        df = open('dataset/pdbbind/index/INDEX_{}_data.2016'.format(self.dataset), 'r').readlines()[6:]
        train_pdb = [i.split()[0] for i in df]
        if split == 'general-core':
            test_pdb = open('dataset/pdbbind/index/core_2016.txt', 'r').read().split()
            train_pdb = list(set(train_pdb) - set(test_pdb))
            random.shuffle(train_pdb)
            val_pdb = train_pdb[:int(len(train_pdb) * 0.2)]
            train_pdb = list(set(train_pdb) - set(val_pdb))
        elif split == 'refined-core':
            test_pdb = open('dataset/pdbbind/index/core_2016.txt', 'r').read().split()
            train_pdb = open('dataset/pdbbind/index/refined_2016.txt', 'r').read().split()
            val_pdb = train_pdb[:int(len(train_pdb) * 0.2)]
            train_pdb = list(set(train_pdb) - set(val_pdb))
        elif split == 'general-refined1000-core':
            test_pdb = open('dataset/pdbbind/index/core_2016.txt', 'r').read().split()
            val_pdb = open('dataset/pdbbind/index/refined_2016.txt', 'r').read().split()
            random.shuffle(val_pdb)
            val_pdb = val_pdb[:500]
            train_pdb = list(set(train_pdb) - set(val_pdb) - set(test_pdb))
        else:  # random split
            random.shuffle(train_pdb)
            val_pdb = train_pdb[int(len(train_pdb) * 0.8):int(len(train_pdb) * 0.9)]
            test_pdb = train_pdb[int(len(train_pdb) * 0.9):]
            train_pdb = train_pdb[:int(len(train_pdb)) * 0.8]

        train_idx, val_idx, test_idx = [], [], []
        idx_dict = np.load(self.dict_files, allow_pickle=True).item()

        for k in idx_dict.keys():
            if k in test_pdb:
                test_idx.append(idx_dict[k])
            elif k in val_pdb:
                val_idx.append(idx_dict[k])
            elif k in train_pdb:
                train_idx.append(idx_dict[k])
        return train_idx, val_idx, test_idx


class mol_data(InMemoryDataset):
    def __init__(self, root='sample', smiles_list=None, transform=None, pre_transform=None, pre_filter=None):
        self.smiles_list = smiles_list
        self.root = root
        self.index = None

        super(mol_data, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = self.process()

    def process(self):
        self.index, data_list = [], []
        for i, smiles in enumerate(self.smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            data = process_mol(mol)
            data_list.append(data)
            self.index.append(i)
        data, slices = self.collate(data_list)
        return data, slices

    @property
    def processed_dir(self):
        return self.root

    @property
    def processed_file_names(self):
        return ['dummy_file']
