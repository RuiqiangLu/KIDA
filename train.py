import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='dataset/pdbbind', type=str, help='dataset root path')
parser.add_argument('--dataset', type=str, default='general_PL', help='refined, general_PL')
parser.add_argument('--split', type=str, default='general-refined1000-core', help='general-refined1000-core, general-core, refined-core')
parser.add_argument('--radius', type=int, default=6, help='4, 6, 0 is knn')
parser.add_argument('--seed', type=int, default=8908)
parser.add_argument('--cuda_ids', default='0', type=str, help='cuda device ids')
parser.add_argument('--save_id', default=0, type=int)
parser.add_argument('--batch_size', default=3, type=int, help='number of batch_size')
parser.add_argument('--accumulation', default=1, type=int, help='gradient accumulation')
parser.add_argument('--epochs', default=50, type=int, help='number of epochs')
parser.add_argument('--loss', default='huber', type=str, help='ce,wce,focal,bfocal...')
parser.add_argument('--optim', default='Adam', type=str, help='Adam, SGD, RMSprop')
parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
parser.add_argument('--checkpoint_dir', default='checkpoint/', type=str)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_ids
import torch
from model import DTI as Mymodel
from utils import seed_torch
from trainer import RegressionTrainer as Trainer
from dataset import PDBBind_Dataset

torch.set_num_threads(6)
seed_torch(args.seed)
print('Loading dataset...')
dataset = PDBBind_Dataset(root=args.dataset_root,
                          dataset=args.dataset,
                          split=args.split)

train_idx, valid_idx, test_idx = dataset.train_idx, dataset.val_idx, dataset.test_idx
train_dataset, valid_dataset, test_dataset = dataset[train_idx], dataset[valid_idx], dataset[test_idx]
print('Training init...')

model = Mymodel(mol_in_dim=16,
                pro_in_dim=15,
                hid_dim=64,
                heads=16,
                num_layers=3,
                )

trainer = Trainer(args, model,
                  train_dataset,
                  valid_dataset,
                  test_dataset,
                  batch_size=args.batch_size // args.accumulation,
                  accumulation_steps=args.accumulation,
                  num_workers=0,
                  )

trainer.train(load_model_id=None)
trainer.test('rmse')
trainer.test('pearson')
