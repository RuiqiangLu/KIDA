from os.path import join
from tqdm import tqdm
import random
import numpy as np
import torch
import time
from torch.optim import Adam, SGD, lr_scheduler, RMSprop
from torch_geometric.loader import DataLoader
from utils import regression_metrics

torch.backends.cudnn.enabled = True
save_id = random.randint(0, 1000)


class Trainer:
    def __init__(self, args, model, train_dataset, valid_dataset, test_dataset,
                 batch_size=8, accumulation_steps=1, num_workers=4):
        self.args = args
        self.model = model.cuda()
        self.save_id = save_id
        self.accumulation_steps = accumulation_steps
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True) if train_dataset is not None else None
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers) if valid_dataset is not None else None
        self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers) if test_dataset is not None else None
        self.save_dir = args.checkpoint_dir
        self.supervise_interaction = False
        self.optimizer = Adam(self.model.parameters(), lr=args.lr)
        self.train_output, self.val_output, self.test_output = None, None, None
        print('\t{}:{}\n'.format(k, v) for k, v in args.__dict__.items())
        print('save id: {}'.format(self.save_id))
        print('train:{}    valid:{}    test: {}'.format(
            len(train_dataset) if train_dataset is not None else 0,
            len(valid_dataset) if valid_dataset is not None else 0,
            len(test_dataset) if test_dataset is not None else 0))

        print("total parameters:" + str(sum([p.nelement() for p in self.model.parameters()])))

    def test(self, index):
        print('best model of validation {}:'.format(index))
        self.load_best_ckpt(index)
        val_output = self.valid_iterations()
        val_loss, val_result = val_output['loss'], val_output['result']
        test_output = self.test_iterations(index=index)
        test_loss, test_result = test_output['loss'], test_output['result']
        loss_info = {'testloss': test_loss.item(), 'valloss': val_loss.item()}
        print('loss: {}\nval_result: {}\ntest_result: {}'.format(loss_info, val_result, test_result))

    def load_and_test(self, load_id, index):
        self.load_ckpt(join(self.save_dir, '{}_{}_model.ckpt'.format(load_id, index)))
        test_output = self.test_iterations(index=index)
        test_result = test_output['result']
        print('test_result: {}'.format(test_result))

    def save_ckpt(self, index):
        file_name = '{}_{}_model.ckpt'.format(self.save_id, index)
        with open(join(self.save_dir, file_name), 'wb') as f:
            torch.save({
                'args': self.args,
                'records': self.records,
                'model_state_dict': self.model.state_dict(), }, f)

    def load_best_ckpt(self, index):
        ckpt_path = join(self.save_dir, '{}_{}_model.ckpt'.format(self.save_id, index))
        self.load_ckpt(ckpt_path)

    def load_ckpt(self, ckpt_path):
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt['model_state_dict'])

    def to_cuda(self, data):
        data.batch = data.batch.cuda()
        data.x = data.x.cuda()
        data.y = data.y.cuda()
        data.edge_attr = data.edge_attr.cuda()
        data.edge_index = data.edge_index.cuda()
        data.mol_node_num = data.mol_node_num.cuda()
        data.pro = data.pro.cuda()
        data.pro_edge_index = data.pro_edge_index.cuda()
        data.pro_edge_attr = data.pro_edge_attr.cuda()
        data.pro_node_num = data.pro_node_num.cuda()
        data.qb_edge_index = data.qb_edge_index.cuda()
        data.qb_edge_attr = data.qb_edge_attr.cuda()
        data.interaction_edge_index = data.interaction_edge_index.cuda()
        data.interaction_edge_attr = data.interaction_edge_attr.cuda()
        data.interaction_edge_num = data.interaction_edge_num.cuda()
        return data

    def data_collation(self, data):
        pro_start, pro_end = 0, 0
        pro_edge_start, pro_edge_end = 0, 0
        qb_edge_start, qb_edge_end = 0, 0
        interaction_edge_start, interaction_edge_end = 0, 0
        new_pro_edge_index = torch.zeros_like(data.pro_edge_index)
        new_qb_edge_index = torch.zeros_like(data.qb_edge_index)
        new_interaction_edge_index = torch.zeros_like(data.interaction_edge_index[1])

        for i in range(data.num_graphs):
            data_i = data[i]
            pro_end += data.pro_node_num[i].item()
            pro_edge_end += data.pro_edge_num[i].item()
            qb_edge_end += data.qb_edge_num[i].item()
            interaction_edge_end += data.interaction_edge_num[i].item()

            data.pro_node[pro_start:pro_end] = i
            new_pro_edge_index[:, pro_edge_start:pro_edge_end] = data_i.pro_edge_index + pro_start
            new_qb_edge_index[:, qb_edge_start:qb_edge_end] = data_i.qb_edge_index + pro_start
            new_interaction_edge_index[interaction_edge_start:interaction_edge_end] = data_i.interaction_edge_index[1] + pro_start
            pro_start = pro_end
            pro_edge_start = pro_edge_end
            qb_edge_start = qb_edge_end
            interaction_edge_start = interaction_edge_end

        data.pro_edge_index = new_pro_edge_index
        data.qb_edge_index = new_qb_edge_index
        data.interaction_edge_index[1] = new_interaction_edge_index

        return data

    def train(self, **kwargs):
        raise NotImplementedError

    def train_iterations(self, **kwargs):
        raise NotImplementedError

    def valid_iterations(self, **kwargs):
        raise NotImplementedError

    def test_iterations(self, **kwargs):
        raise NotImplementedError

    def log_and_save(self, **kwargs):
        raise NotImplementedError

    def save_best_ckpt(self, max_best, min_best):
        for item in max_best:
            if max_best[item] == np.array(self.records[item]).max():
                self.save_ckpt(item)
                print('save best validation {} {}'.format(item, max_best[item]))
        for item in min_best:
            if min_best[item] == np.array(self.records[item]).min():
                self.save_ckpt(item)
                print('save best validation {} {}'.format(item, min_best[item]))


class RegressionTrainer(Trainer):
    def __init__(self, args, model, train_dataset, valid_dataset, test_dataset,
                 batch_size=8, accumulation_steps=1, num_workers=4):
        super(RegressionTrainer, self).__init__(args, model, train_dataset, valid_dataset, test_dataset,
                                                batch_size=batch_size,
                                                accumulation_steps=accumulation_steps,
                                                num_workers=num_workers)
        self.records = {'val_losses': [], 'pearson': [], 'rmse': []}
        self.metrics_fn = regression_metrics
        self.pretrain_index = 'complex_pearson'

    def train(self, load_model_id=None):
        print('Training start...')
        for epoch in tqdm(range(self.args.epochs)):
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            self.train_output = self.train_iterations()
            self.val_output = self.valid_iterations()
            self.test_output = self.valid_iterations('test')
            self.log_and_save(epoch)

    def train_iterations(self):
        self.model.train()
        losses, ys_true, ys_pred = [], [], []
        for i, data in enumerate(self.train_dataloader):
            data = self.data_collation(data)
            mol_batch = self.to_cuda(data)
            output = self.model(mol_batch, y_true=data.y.unsqueeze(1))
            loss = output['loss']
            loss = loss / self.accumulation_steps
            loss.backward()
            if (i + 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            losses.append(loss.item())
        loss = np.array(losses).mean()

        return {'loss': loss}

    @torch.no_grad()
    def valid_iterations(self, dataset='valid'):
        self.model.eval()
        if dataset == 'valid':
            dataloader = self.valid_dataloader
        elif dataset == 'test':
            dataloader = self.test_dataloader
        losses, ys_true, ys_pred = [], [], []
        for data in dataloader:
            data = self.data_collation(data)
            mol_batch = self.to_cuda(data)
            output = self.model(mol_batch, y_true=data.y.unsqueeze(1))
            y_pred = output['y_pred']
            loss = output['loss']
            losses.append(loss.item())
            ys_true.append(data.y.unsqueeze(1).cpu())
            ys_pred.append(y_pred.cpu())
        loss = torch.tensor(losses).mean()
        result = self.metrics_fn(torch.cat(ys_true).numpy(), torch.cat(ys_pred).numpy())
        return {'loss': loss,
                'result': result}

    @torch.no_grad()
    def test_iterations(self, index='rmse'):
        self.model.eval()
        dataloader = self.test_dataloader
        losses, ys_true, ys_pred = [], [], []
        for data in dataloader:
            data = self.data_collation(data)
            mol_batch = self.to_cuda(data)
            y_true = data.y.unsqueeze(1)
            output = self.model(mol_batch, y_true=y_true)
            y_pred = output['y_pred']
            loss = output['loss']
            losses.append(loss.cpu())
            ys_true.append(y_true.cpu())
            ys_pred.append(y_pred.cpu())
        loss = torch.tensor(losses).mean()
        result = self.metrics_fn(torch.cat(ys_true).numpy(), torch.cat(ys_pred).numpy())

        return {'loss': loss, 'result': result}

    def log_and_save(self, epoch):
        train_loss = self.train_output['loss']
        val_loss = self.val_output['loss']
        val_result = self.val_output['result']

        print('Epoch: {}, train_loss: {:.4f}, val_loss: {:.4f}  rmse:{:.4f} pearson:{:.4f}, mae:{:.4f}'.format(
                epoch, train_loss, val_loss,  val_result['rmse'], val_result['pearson'], val_result['mae']))

        self.records['val_losses'].append(val_loss)
        self.records['rmse'].append(val_result['rmse'])
        self.records['pearson'].append(val_result['pearson'])

        max_best = {'pearson': val_result['pearson']}
        min_best = {'rmse': val_result['rmse']}
        self.save_best_ckpt(max_best, min_best)
