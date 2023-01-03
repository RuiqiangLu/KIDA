import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import Block, GraphAggregator
from torch_geometric.utils import to_dense_batch, to_dense_adj
from EGNN import EGNN_Sparse
from torch_scatter import scatter_max, scatter_mean, scatter_sum
from utils import mdn_loss_fn


def create_batch(sizes):
    sizes = sizes.tolist()
    sizes = list(map(int, sizes))
    batch = []
    for i, size in enumerate(sizes):
        batch.extend([i] * size)
    batch = torch.tensor(batch, dtype=torch.int64).cuda()
    return batch


def create_index(size_1, size_2):
    size_1, size_2 = size_1.tolist(), size_2.tolist()
    batch_size = len(size_1)
    index_1, index_2 = [], []
    ptr_1, ptr_2 = 0, 0
    for i in range(batch_size):
        for j in range(size_1[i]):
            index_1.extend([j + ptr_1] * size_2[i])
        index_2.extend(list(range(ptr_2, size_2[i] + ptr_2)) * size_1[i])
        ptr_1 += size_1[i]
        ptr_2 += size_2[i]
    return torch.LongTensor(index_1).cuda(), torch.LongTensor(index_2).cuda()


def vec_to_dense_adj(vec, index_1, index_2):
    dense_adj = to_dense_adj(torch.cat([index_1.unsqueeze(0),
                             (index_1[-1] + index_2 + 1).unsqueeze(0)], dim=0),
                             edge_attr=vec)[0].permute(2, 1, 0)
    dense_adj = dense_adj[:, index_1[-1]+1:, :index_1[-1]+1]
    return dense_adj


class PositionalEncoding(nn.Module):
    def __init__(self, in_dim, hid_dim, num_layers=3, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.transform = nn.Linear(in_dim, hid_dim)
        self.layers = nn.ModuleList()
        # self.residue_layers = nn.ModuleList()
        self.batch_norm = nn.BatchNorm1d(hid_dim)
        for i in range(num_layers):
            self.layers.append(EGNN_Sparse(feats_dim=hid_dim, m_dim=64))

    def forward(self, x, edge_index, batch):
        x = torch.cat([x[:, :3], self.dropout(F.celu(self.transform(x[:, 3:])))], dim=1)
        for layer in self.layers:
            x[:, 3:] = x[:, 3:] + layer(x=x, edge_index=edge_index, batch=batch)[:, 3:]
            x[:, 3:] /= 2.
        x = self.batch_norm(x[:, 3:])
        return x


class GraphEmbedding(nn.Module):
    # Nan in molecular cov
    def __init__(self, in_dim, hid_dim, edge_dim, num_layers):
        super(GraphEmbedding, self).__init__()
        self.transform = nn.Linear(in_dim, hid_dim)
        self.GCNs = nn.ModuleList()
        for i in range(num_layers):
            self.GCNs.append(Block(hid_dim, edge_dim))

    def forward(self, x, edge_index, edge_attr):
        x = F.celu(self.transform(x))
        for layer in self.GCNs:
            x = layer(x, edge_index, edge_attr)
        return x


class ComplexPredictor(nn.Module):
    def __init__(self, hid_dim=32, heads=8, output=1, dropout=0.1):
        super(ComplexPredictor, self).__init__()
        self.heads = heads
        mol_dim = pro_dim = hid_dim
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(mol_dim+pro_dim, heads)
        # self.linear2 = nn.Linear(16*heads, heads)
        self.linear3 = nn.Linear(heads, output)

    def forward(self, mol_feats, pro_feats, pro_batch, bipartite_edge_index, bipartite_edge_attr):

        dist = bipartite_edge_attr[:, 0].unsqueeze(1)
        x = dist * torch.cat([mol_feats[bipartite_edge_index[0]], pro_feats[bipartite_edge_index[1]]], dim=1)
        mol_num = mol_feats.shape[0]

        x = self.dropout(F.celu(self.linear(x)) + 1)
        interaction_mat = to_dense_adj(torch.cat([bipartite_edge_index[0].unsqueeze(0),
                                                  bipartite_edge_index[1].unsqueeze(0) + mol_num]),
                                                  edge_attr=x)[0, :mol_num, mol_num:].permute(2, 0, 1)
        padding = pro_feats.shape[0] - interaction_mat.shape[-1]
        if padding > 0:
            interaction_mat = torch.cat([interaction_mat,
                                         torch.zeros([self.heads, interaction_mat.shape[-2], padding],
                                                     dtype=torch.float, device=interaction_mat.device)], dim=-1)

        y_complex_pred = torch.sum(interaction_mat, dim=1)
        y_complex_pred = torch.cat(
            [to_dense_batch(y_complex_pred[i], pro_batch)[0].unsqueeze(0) for i in range(self.heads)], dim=0)

        y_complex_pred = torch.sum(y_complex_pred * 0.01, dim=-1).permute(1, 0)
        # y_complex_pred = torch.celu(self.linear(y_complex_pred))
        y_complex_pred = self.linear3(y_complex_pred)

        return interaction_mat, y_complex_pred


class ComplexFreePredictor(nn.Module):
    def __init__(self, hid_dim=32, heads=8, output=1, dropout=0.1):
        super(ComplexFreePredictor, self).__init__()
        # self.pi = nn.Linear(hid_dim * 2, heads)
        self.sigma = nn.Linear(hid_dim * 2, heads)
        self.mu = nn.Linear(hid_dim * 2, heads)

        self.linear1 = nn.Linear(heads, heads*2)
        self.linear2 = nn.Linear(heads*2, output)

    def forward(self, mol_feats, pro_feats, spatial_feats, mol_size, pro_size, mol_batch):
        pro_feats = pro_feats * spatial_feats

        mol_index, pro_index = create_index(mol_size, pro_size)
        atom_pairs = torch.cat([mol_feats[mol_index], pro_feats[pro_index]], dim=-1)

        # pi = F.softmax(self.pi(atom_pairs), dim=-1) + 0.0001
        sigma = F.elu(self.sigma(atom_pairs)) + 1.1
        mu = F.elu(self.mu(atom_pairs)) + 1

        # mu = self.topN(mu, mol_index, pro_index)

        y_pred = scatter_sum(mu, index=mol_index, dim=0)
        y_pred = scatter_sum(y_pred, index=mol_batch, dim=0) * 0.001

        y_pred = F.elu(self.linear1(y_pred))
        y_pred = self.linear2(y_pred)

        return mu, sigma, mol_index, pro_index, y_pred

    def topN(self, mu, mol_index, pro_index):
        mu = vec_to_dense_adj(mu, mol_index, pro_index).permute(0, 2, 1)
        # _, indices1 = mu.topk(50, dim=-1, largest=True)
        # _, indices2 = mu.topk(10, dim=-2, largest=True)
        # topk = torch.logical_and(torch.zeros_like(mu, dtype=torch.bool).scatter_(-1, indices1, True),
        #                          torch.zeros_like(mu, dtype=torch.bool).scatter_(-2, indices2, True))
        # mu = torch.where(topk, mu, torch.tensor(0, dtype=mu.dtype, device=mu.device))
        mu = mu[:, mol_index, pro_index].permute(1, 0)
        return mu


class DTI(nn.Module):
    def __init__(self,
                 mol_in_dim=16,
                 mol_edge_dim=4,
                 pro_in_dim=15,
                 pro_edge_dim=1,
                 hid_dim=64,
                 heads=8,
                 num_layers=3,
                 dropout=0.1,
                 output=1,
                 binary=False,
                 local_rank=None,
                 load_model=None,):
        super(DTI, self).__init__()

        self.mol_embedding = GraphEmbedding(mol_in_dim, hid_dim, mol_edge_dim, num_layers=num_layers)
        self.pro_embedding = GraphEmbedding(pro_in_dim, hid_dim, pro_edge_dim, num_layers=num_layers)
        self.position_encode = PositionalEncoding(in_dim=pro_in_dim, hid_dim=hid_dim, num_layers=num_layers)

        self.complex_predictor = ComplexPredictor(hid_dim=hid_dim, heads=heads, output=output, dropout=dropout)
        self.complex_free_predictor = ComplexFreePredictor(hid_dim=hid_dim, heads=heads, output=output, dropout=dropout)

        self.loss = modified_huber_loss if binary else torch.nn.SmoothL1Loss()
        self.loss = self.loss if local_rank is None else self.loss.cuda(local_rank)
        # self.mdn_loss = mdn_loss_fn if local_rank is None else mdn_loss_fn.cuda(local_rank)

    def forward(self, data, y_true, mode=3):
        mol_feats, mol_edge_index, mol_edge_attr, mol_size = data.x, data.edge_index, data.edge_attr, data.mol_node_num
        pro_feats, pro_edge_index, pro_edge_attr, pro_size = data.pro, data.pro_edge_index, data.pro_edge_attr, data.pro_node_num
        knn_index = data.qb_edge_index
        bipartite_edge_index, bipartite_edge_attr = data.interaction_edge_index, data.interaction_edge_attr
        pro_batch = create_batch(pro_size)

        if mode == 1:   # complex_base
            mol_feats = self.mol_embedding(mol_feats, mol_edge_index, mol_edge_attr)
            pro_feats = self.pro_embedding(pro_feats[:, 3:], pro_edge_index, pro_edge_attr)
            _, y_complex_pred = self.complex_predictor(mol_feats, pro_feats, pro_batch, bipartite_edge_index, bipartite_edge_attr)
            loss = self.loss(y_complex_pred, y_true)
            return {'y_complex_pred': y_complex_pred,
                    'y_pred': torch.zeros_like(y_complex_pred, device='cpu'),
                    'interaction_loss': torch.tensor([0.]),
                    'loss': loss,
                    'pred_loss': torch.tensor([0.]),
                    'complex_pred_loss': loss}

        elif mode == 2:  # complex_free
            mol_feats = self.mol_embedding(mol_feats, mol_edge_index, mol_edge_attr)
            spatial_feats = self.position_encode(pro_feats, knn_index, pro_batch)
            pro_feats = self.pro_embedding(pro_feats[:, 3:], pro_edge_index, pro_edge_attr)
            _, _, _, _, _, y_pred = self.complex_free_predictor(mol_feats, pro_feats, spatial_feats, mol_size, pro_size, data.batch)
            loss = self.loss(y_pred, y_true)
            return {'y_complex_pred': torch.zeros_like(y_pred, device='cpu'),
                    'y_pred': y_pred,
                    'interaction_loss': torch.tensor([0.]),
                    'loss': loss,
                    'pred_loss': loss,
                    'complex_pred_loss': torch.tensor([0.])}

        else:   # complex_base and complex_free
            mol_feats = self.mol_embedding(mol_feats, mol_edge_index, mol_edge_attr)
            spatial_feats = self.position_encode(pro_feats, knn_index, pro_batch)
            pro_feats = self.pro_embedding(pro_feats[:, 3:], pro_edge_index, pro_edge_attr)
            interaction_mat, y_complex_pred = self.complex_predictor(mol_feats, pro_feats, pro_batch,
                                                                     bipartite_edge_index, bipartite_edge_attr)
            mu, sigma, mol_index, pro_index, y_pred = self.complex_free_predictor(mol_feats, pro_feats,
                                                                                      spatial_feats, mol_size,
                                                                                      pro_size, data.batch)
            loss = self.loss(y_pred, y_true)
            complex_loss = self.loss(y_complex_pred, y_true)
            interaction_loss = torch.mean(mdn_loss_fn(sigma, mu, interaction_mat[:, mol_index, pro_index].permute(1, 0)))

            return {'y_complex_pred': y_complex_pred,
                    'y_pred': y_pred,
                    # 'interaction_loss': torch.tensor([0.]),
                    'interaction_loss': interaction_loss,
                    'loss': loss + complex_loss + interaction_loss,
                    'pred_loss': loss,
                    'complex_pred_loss': complex_loss}


def modified_huber_loss(pred, true):
    true = torch.where(true == 1., 1., -1.)
    condition_1 = pred * true
    condition_2 = 1 - 2*condition_1 + condition_1*condition_1
    result = torch.where(condition_2 > 0., condition_2, torch.zeros_like(condition_2))
    result = torch.where(condition_1 < -1, -4 * condition_1, result)
    return torch.sum(result)


# def modified_huber_loss2(pred, true, decay=0.618):
#     true = torch.where(true == 1., 1., -1.)
#     pred = pred - (1 - true)*decay
#     condition_1 = pred * true
#     condition_2 = 1 - 2*condition_1 + condition_1*condition_1
#     result = torch.where(condition_2 > 0., condition_2, torch.zeros_like(condition_2))
#     result = torch.where(condition_1 < -1, -4 * condition_1, result)
#     return torch.sum(result)
