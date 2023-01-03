import torch
import numpy as np
from scipy.sparse import csr_matrix, spmatrix
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType, BondType
from torch_scatter import scatter
from torch_geometric.data import Data
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.edges.atomic import add_atomic_edges
from graphein.protein.edges.distance import add_distance_threshold
from graphein.protein.graphs import construct_graph
from graphein.ml.conversion import GraphFormatConvertor
from plip.basic import config as plip_config
from plip.structure.preparation import PDBComplex
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from functools import partial


plip_config.NOFIXFILE = True
plip_config.NOHYDRO = True


nx_to_pyg = GraphFormatConvertor(src_format='nx', dst_format='pyg')
atom_graph_param = {"granularity": "atom",
                    "edge_construction_functions": [add_atomic_edges],
                    "verbose": False}
atom_graph_config = ProteinGraphConfig(**atom_graph_param)
residue_graph_param = {"granularity": "CA",
                       "edge_construction_functions": [partial(add_distance_threshold,
                                                       long_interaction_threshold=0,
                                                       threshold=8)],
                       "verbose": False}
residue_graph_config = ProteinGraphConfig(**residue_graph_param)


def create_complex(mol, pocket_str):
    mol_str = Chem.rdmolfiles.MolToPDBBlock(mol).split('\n')[1:]
    complex_str = ''.join(pocket_str) + '\n'.join(mol_str)
    return complex_str


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        pass
    return list(map(lambda s: x == s, allowable_set))


def get_mol_nodes_edges(mol, get_coords=True):
    # Read node features
    atom_coord = []
    N = mol.GetNumAtoms()
    atom_type = []
    atomic_number = []
    aromatic = []
    hybridization = []
    # num_hs = []
    for i, atom in enumerate(mol.GetAtoms()):
        atom_type.append(atom.GetSymbol())
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization.append(atom.GetHybridization())
    if get_coords:
        for i, atom in enumerate(mol.GetAtoms()):
            atom_coord.append([mol.GetConformer().GetAtomPosition(i).x,
                               mol.GetConformer().GetAtomPosition(i).y,
                               mol.GetConformer().GetAtomPosition(i).z])

    # Read edge features
    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bond.GetBondType()]
    edge_index = torch.LongTensor([row, col])
    edge_type = [one_of_k_encoding(t, [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC])
                 for t in edge_type]
    edge_attr = torch.FloatTensor(edge_type)
    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_attr = edge_attr[perm]
    row, col = edge_index

    # Concat node features
    hs = (torch.tensor(atomic_number, dtype=torch.long) == 1).to(torch.float)
    num_hs = scatter(hs[row], col, dim_size=N).tolist()
    x_atom_type = [one_of_k_encoding(t, ['H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I', 'P']) for t in atom_type]
    x_hybridization = [one_of_k_encoding(h, [HybridizationType.SP, HybridizationType.SP2, HybridizationType.SP3])
                       for h in hybridization]
    x2 = torch.tensor([atomic_number, aromatic, num_hs], dtype=torch.float).t().contiguous()
    x = torch.cat([torch.FloatTensor(x_atom_type), torch.FloatTensor(x_hybridization), x2], dim=-1)
    return x, edge_index, edge_attr, x.shape[0], atom_coord


def get_pro_nodes_edge(pdb, chain_residue_index):
    g = construct_graph(config=atom_graph_config, pdb_path=pdb)
    df = g.graph['pdb_df']
    residue_embed_index = [i.split(':')[0] + '_' + i.split(':')[2] for i in g.nodes]
    residue_embed_index = [chain_residue_index.index(i) for i in residue_embed_index]
    # residue_embed = residue_embed[residue_embed_index]
    atoms = g.nodes
    atom_coord = [i[1]['coords'] for i in atoms.data()]
    atoms_type_dict = {}
    atom_element = df['element_symbol']
    for i in range(len(df)):
        if atom_element[i] == 'H':
            atoms_type_dict[df['node_id'][i]] = 'Hsb'
        else:
            atoms_type_dict[df['node_id'][i]] = df['atom_bond_state'][i]
    atoms_type = []
    for atom in atoms:
        atoms_type.append(atoms_type_dict[atom])
    bond_hybrid_dict = {'Csb': 'sp3', 'Cdb': 'sp2', 'Cres': 'sp2',
                        'Osb': 'sp3', 'Odb': 'sp2', 'Ores': 'sp2',
                        'Nsb': 'sp3', 'Ndb': 'sp2', 'Nres': 'sp2',
                        'Hsb': 'other', 'Ssb': 'other'}   # https://arxiv.org/pdf/0804.2488
    atomic_dict = {'C': 6, 'O': 8, 'N': 7, 'H': 1, 'S': 16}
    atom_hybridization = [bond_hybrid_dict[i] for i in atoms_type]
    x_atom_type = [one_of_k_encoding(i[0], ['H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I', 'P']) for i in atoms_type]
    x_hybridization = [one_of_k_encoding(h, ['sp', 'sp2', 'sp3']) for h in atom_hybridization]
    x_atomic_number = [atomic_dict[i[0]] for i in atoms_type]
    x_aromatic = [i[1:] == 'res' for i in atoms_type]
    x2 = torch.tensor([x_atomic_number, x_aromatic], dtype=torch.float).t().contiguous()
    x = torch.cat([torch.FloatTensor(atom_coord), torch.FloatTensor(x_atom_type), torch.FloatTensor(x_hybridization), x2], dim=-1)
    pyg_g = nx_to_pyg.convert_nx_to_pyg(g)
    pro_node = torch.tensor(len(g.nodes)).long()
    edge_attr = []
    for i in range(pyg_g.edge_index.shape[1]):
        node1, node2 = pyg_g.node_id[pyg_g.edge_index[0][i]], pyg_g.node_id[pyg_g.edge_index[1][i]]
        edge_attr.append(g.adj[node1][node2]['bond_length'])
    edge_attr = torch.FloatTensor([[i] for i in edge_attr])
    residue_embed_index = torch.tensor(residue_embed_index, dtype=torch.long).reshape(-1)
    return Data(x=x,
                edge_index=pyg_g.edge_index,
                edge_attr=edge_attr,
                pro_node_num=len(g.nodes),
                pro_node=pro_node,
                residue_pro_map=residue_embed_index), atom_coord


def get_bipartite_graph(lig_coord, pro_coord):
    return


def query_ball_graph(xyz, distance=4., knn=False):
    if knn:
        neighbors = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(xyz)
        adj = neighbors.kneighbors_graph(xyz, mode='distance').toarray()
    else:
        distance_map = pairwise_distances(np.vstack(xyz))
        adj = np.where(distance_map < distance, distance_map, 0)

    coo_adj = spmatrix.tocoo(csr_matrix(adj))
    edge_index = torch.LongTensor(np.vstack([coo_adj.row, coo_adj.col]))
    edge_attr = torch.FloatTensor(coo_adj.data)

    return Data(edge_index=edge_index, edge_attr=edge_attr)


def get_contact_map(pdb):
    g = construct_graph(config=residue_graph_config, pdb_path=pdb)
    chain_ids, residue_numbers = g.graph['pdb_df']['chain_id'], g.graph['pdb_df']['residue_number']
    chain_residue_index = [chain_ids[i] + '_' + str(residue_numbers[i]) for i in range(len(g.graph['pdb_df']))]
    return None, chain_residue_index


def mol2data(mol, protein, pro_atom_coord, pdbid=None, complex_str=None, label=0):
    mol, mol_edge_index, mol_edge_attr, mol_node_num, mol_atom_coord = get_mol_nodes_edges(mol)
    interaction = get_interaction_graph(mol_atom_coord, pro_atom_coord, pdbid=pdbid, pdbstr=complex_str)
    if type(label) is int:
        y = torch.LongTensor([label])
    elif type(label) is float:
        y = torch.FloatTensor([label])
    else:
        print('error label type')
        return None
    data = Data(x=mol,
                edge_index=mol_edge_index,
                edge_attr=mol_edge_attr,
                y=y,
                pdbid=protein.pdbid,
                pro=protein.x,
                pro_edge_index=protein.edge_index,
                pro_edge_attr=protein.edge_attr,
                pro_node_num=protein.pro_node_num,
                pro_edge_num=protein.edge_attr.shape[0],
                pro_node=protein.pro_node,
                mol_node_num=mol_node_num,
                interaction_edge_index=interaction.edge_index,
                interaction_edge_attr=interaction.edge_attr,
                interaction_edge_num=interaction.edge_attr.shape[0],
                qb_edge_index=protein.qb_edge_index,
                qb_edge_attr=protein.qb_edge_attr,
                qb_edge_num=protein.qb_edge_num,
                )
    return data
