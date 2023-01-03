import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from model import DTI as Mymodel
from trainer import RegressionTrainer as Trainer
from dataset import PDBBind_Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='dataset/pdbbind', type=str, help='dataset root path')
parser.add_argument('--dataset', type=str, default='general_PL', help='general_PL')
parser.add_argument('--split', type=str, default='general-core', help='random, scaffold')
parser.add_argument('--radius', type=int, default=6, help='4, 6')
parser.add_argument('--batch_size', default=1, type=int, help='number of batch_size')
parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
parser.add_argument('--checkpoint_dir', default='checkpoint/', type=str)

args = parser.parse_args()
print('Loading dataset...')
dataset = PDBBind_Dataset(root=args.dataset_root,
                          dataset=args.dataset,
                          split=args.split,)

train_idx, valid_idx, test_idx = dataset.train_idx, dataset.val_idx, dataset.test_idx
train_dataset, valid_dataset, test_dataset = dataset[train_idx], dataset[valid_idx], dataset[test_idx]
print('Testing init...')
with torch.no_grad():
    model = Mymodel(mol_in_dim=16,
                    pro_in_dim=15,
                    hid_dim=64,
                    heads=16,
                    num_layers=3,
                    )

    trainer = Trainer(args,
                      model,
                      train_dataset,
                      valid_dataset,
                      test_dataset,
                      batch_size=args.batch_size,
                      num_workers=0)

trainer.load_and_test(389, 'pearson')


