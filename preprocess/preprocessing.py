import os
import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from preprocess_utils import get_contact_map, get_pro_nodes_edge, query_ball_graph, mol2data
from multiprocessing import Pool


def save_data(pdbid, idx, y):
    try:
        root = '../dataset/pdbbind'
        dataset_dir = os.path.join(root, 'total-set', pdbid)
        sdf_file = os.path.join(dataset_dir, '{}_fixed.sdf'.format(pdbid))
        pocket = os.path.join(dataset_dir, '{}_pocket_6A.pdb'.format(pdbid))
        pdb = os.path.join(dataset_dir, '{}_fix.pdb'.format(pdbid))
        mol = Chem.SDMolSupplier(sdf_file)[0]
        if mol.GetNumAtoms() < 10:
            return None, None
        residue_graph, chain_residue_index = get_contact_map(pdb)
        protein, pro_atom_coord = get_pro_nodes_edge(pocket, chain_residue_index)
        qb_graph = query_ball_graph(pro_atom_coord)
        protein = Data(x=protein.x,
                       edge_index=protein.edge_index,
                       edge_attr=protein.edge_attr,
                       pro_node_num=protein.pro_node_num,
                       pro_node=protein.pro_node,
                       qb_edge_index=qb_graph.edge_index,
                       qb_edge_attr=qb_graph.edge_attr,
                       qb_edge_num=qb_graph.edge_attr.shape[0],
                       pdbid=pdbid)
        data = mol2data(mol, protein, pro_atom_coord, pdbid=pdbid, label=y)
        data_pt = os.path.join(root, 'processed/{}.pt'.format(idx))
        torch.save(data, data_pt)
    except (RuntimeError, ValueError, AttributeError) as e:
        print(pdbid, e)
        return None, None
    return pdbid, idx


if __name__ == '__main__':
    root = '../dataset/pdbbind'
    df = open(root + '/index/INDEX_general_PL_data.2016', 'r').readlines()[6:]
    pdbbind_list = [i.split()[0] for i in df]
    labels = [i.split()[3] for i in df]
    s = list(zip(pdbbind_list, labels))
    pdbbind_list, labels = zip(*s)
    ys = []
    dataset_dict = {}
    # dataset_root = '../dataset/pdbbind/'
    raw_dir = os.path.join(root, 'total-set')
    pdbids = os.listdir(raw_dir)
    start = 0
    for idx, pdbid in enumerate(pdbbind_list):
        sdf_file = os.path.join(raw_dir, pdbid, '{}_fixed.sdf'.format(pdbid))
        if pdbid in pdbbind_list and os.path.exists(sdf_file):
            if Chem.SDMolSupplier(sdf_file)[0] is not None:
                dataset_dict[pdbid] = idx
                ys.append(float(labels[idx]))

    pool = Pool(4)
    result = []
    for i, pdbid in enumerate(list(dataset_dict.keys())):
        idx = dataset_dict[pdbid]
        result.append(pool.apply_async(func=save_data, args=(pdbid, idx, ys[i])))
    pool.close()
    pool.join()

    new_dataset_dict = {}
    for i in result:
        pdbid, idx = i.get()
        new_dataset_dict[pdbid] = idx
    element = new_dataset_dict.pop(None)
    np.save(root + 'idx_dict.npy', new_dataset_dict)
    print('finished !!!')

