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
    print(pdbid)
    return pdbid, idx


def process_pdb(pdb, pocket):
    # pocket = os.path.join(pocket_file)
    # pdb = os.path.join(pdb_file)
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
                   )
    return protein




if __name__ == '__main__':

    # core_2016_pdbids = '''1a30  1h22  1o3f  1q8u  1vso  2al5  2iwx  2qe4  2w66  2x00  2yfe  3acw  3b27  3d4z  3ebp  3fv1  3gnw  3kgp  3n86  3p5o  3rr4  3u9q  3uri  4abg  4crc  4e5w  4f9w  4ivc  4k18  4mgd  4twp  5a7b
    # 1bcu  1h23  1o5b  1qf1  1w4o  2br1  2j78  2qnq  2wbg  2xb8  2yge  3ag9  3b5r  3d6q  3ehy  3fv2  3gr2  3kr8  3nq9  3prs  3rsx  3udh  3utu  4agn  4ddh  4e6q  4gfm  4ivd  4k77  4mme  4ty7  5aba
    # 1bzc  1k1i  1owh  1qkt  1y6r  2brb  2j7h  2r9w  2wca  2xbv  2yki  3ao4  3b65  3dd0  3ejr  3g0w  3gv9  3kwa  3nw9  3pww  3ryj  3ueu  3uuo  4agp  4ddk  4ea2  4gid  4j21  4kz6  4ogj  4u4s  5c28
    # 1c5z  1lpg  1oyt  1r5y  1yc1  2c3i  2p15  2v00  2weg  2xdl  2ymd  3arp  3b68  3dx1  3f3a  3g2n  3gy4  3l7b  3nx7  3pxf  3syr  3uev  3wtj  4agq  4de1  4eky  4gkm  4j28  4kzq  4owm  4w9c  5c2h
    # 1e66  1mq6  1p1n  1s38  1ydr  2cbv  2p4y  2v7a  2wer  2xii  2zb1  3arq  3bgz  3dx2  3f3c  3g2z  3ivg  3lka  3o9i  3pyy  3tsk  3uew  3wz8  4bkt  4de2  4eo8  4gr0  4j3l  4kzu  4pcs  4w9h  5dwr
    # 1eby  1nc1  1p1q  1sqa  1ydt  2cet  2pog  2vkm  2wn9  2xj7  2zcq  3aru  3bv9  3dxg  3f3d  3g31  3jvr  3mss  3oe4  3qgy  3twp  3uex  3zdg  4cig  4de3  4eor  4hge  4jfs  4llx  4qac  4w9i  5tmn
    # 1g2k  1nc3  1ps3  1syi  1z6e  2fvd  2qbp  2vvn  2wnc  2xnb  2zcr  3arv  3cj4  3e5a  3f3e  3gbb  3jvs  3myg  3oe5  3qqs  3u5j  3ui7  3zso  4ciw  4djv  4f09  4ih5  4jia  4lzs  4qd6  4w9l
    # 1gpk  1nvq  1pxn  1u1b  1z95  2fxs  2qbq  2vw5  2wtv  2xys  2zda  3ary  3coy  3e92  3fcq  3gc5  3jya  3n76  3ozs  3r88  3u8k  3uo4  3zsx  4cr9  4dld  4f2w  4ih7  4jsz  4m0y  4rfm  4wiv
    # 1gpn  1o0h  1q8t  1uto  1z9g  2hb1  2qbr  2w4x  2wvt  2y5h  2zy1  3b1m  3coz  3e93  3fur  3ge7  3k5v  3n7a  3ozt  3rlr  3u8n  3up2  3zt2  4cra  4dli  4f3c  4ivb  4jxs  4m0z  4tmn  4x6p'''
    # core_2016_pdbids = core_2016_pdbids.split()

    root = '../dataset/pdbbind/'
    if not os.path.exists(os.path.join(root, 'processed')):
        os.makedirs(os.path.join(root, 'processed'))
    df = open(root + '/index/INDEX_general_PL_data.2016', 'r').readlines()[6:]
    pdbbind_list = [i.split()[0] for i in df]
    # pdbbind_list = core_2016_pdbids
    labels = [i.split()[3] for i in df]
    s = list(zip(pdbbind_list, labels))
    pdbbind_list, labels = zip(*s)
    ys = []
    dataset_dict = {}
    raw_dir = os.path.join(root, 'total-set')
    pdbids = os.listdir(raw_dir)
    start = 0
    for idx, pdbid in enumerate(pdbbind_list):
        sdf_file = os.path.join(raw_dir, pdbid, '{}_fixed.sdf'.format(pdbid))
        if pdbid in pdbbind_list and os.path.exists(sdf_file):
            if Chem.SDMolSupplier(sdf_file)[0] is not None:
                dataset_dict[pdbid] = idx
                ys.append(float(labels[idx]))

    pool = Pool(20)
    result = []
    for i, pdbid in enumerate(list(dataset_dict.keys())):
        idx = dataset_dict[pdbid]
        result.append(pool.apply_async(func=save_data, args=(pdbid, idx, ys[i])))
    pool.close()
    pool.join()

    print('process_finish')
    new_dataset_dict = {}
    for i in result:
        pdbid, idx = i.get()
        new_dataset_dict[pdbid] = idx
    element = new_dataset_dict.pop(None)
    np.save(root + 'idx_dict.npy', new_dataset_dict)
    print('finished !!!')

