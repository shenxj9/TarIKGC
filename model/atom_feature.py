import numpy as np
import torch
from rdkit import Chem
import networkx as nx
from dgl import DGLGraph
import dgl
from rdkit.Chem import MolFromSmiles, AllChem
from sklearn.metrics import pairwise_distances

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smiles2graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None ,None ,None
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature/sum(feature))
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    return c_size, features, edge_index

def get_graphs_features(compound_smiles):

    h=list()
    graphs=list()
    # print('compound_smiles',compound_smiles)
    for i in range(len(compound_smiles)):
        # print('smiles:',compound_smiles[i])
        c_size, features, edge_index=smiles2graph(compound_smiles[i])
        g=DGLGraph()
        if c_size == None:
            print('compound_smiles:',compound_smiles[i][1])
        g.add_nodes(c_size)
        if edge_index:
            edge_index=np.array(edge_index)
            g.add_edges(edge_index[:,0],edge_index[:,1])

        for f in features:
            h.append(f)
        g.ndata['x']=torch.from_numpy(np.array(features))
        g=dgl.add_self_loop(g)
        graphs.append(g)
    g=dgl.batch(graphs)
    return g,torch.from_numpy(np.array(h,dtype=np.float32))

def smile_to_graph(smile):
    mol = MolFromSmiles(smile)
    try:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol) #, maxAttempts=5000
        AllChem.UFFOptimizeMolecule(mol)
        mol = Chem.RemoveHs(mol)
    except:
        AllChem.Compute2DCoords(mol)
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))
    node_features = np.array(features)
    adj_matrix = np.eye(mol.GetNumAtoms())
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtom().GetIdx()
        end_atom = bond.GetEndAtom().GetIdx()
        adj_matrix[begin_atom, end_atom] = adj_matrix[end_atom, begin_atom] = 1
    conf = mol.GetConformer()
    pos_matrix = np.array([[conf.GetAtomPosition(k).x, conf.GetAtomPosition(k).y, conf.GetAtomPosition(k).z]
                           for k in range(mol.GetNumAtoms())])
    dist_matrix = pairwise_distances(pos_matrix)
    return c_size, node_features, adj_matrix, dist_matrix

def pad_array(array, size):
    result = torch.zeros(size=size)
    slices = tuple(slice(s) for s in array.shape)
    result[slices] = torch.tensor(array)
    return result

def normalize(mx):
    rowsum = np.array(mx.sum(1) + 1e-10)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)  # 此处得到一个对角矩阵
    mx = r_mat_inv.dot(mx)  # 注意.dot为矩阵乘法,不是对应元素相乘
    return mx


def normalize_adj(mx):
    '''D^(-1/2)AD^(-1/2)'''
    np.fill_diagonal(mx, 1)
    d = np.array(mx.sum(1))
    r_inv_sqrt = np.power(d, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = np.diag(r_inv_sqrt)
    mid = np.dot(r_mat_inv_sqrt, mx)
    return np.dot(mid, r_mat_inv_sqrt)
def get_compound_feature(smi):
    cpd_cut_len = 90
    features = smile_to_graph(smi)
    cpd_atom_features, cpd_adj_matrix, cpd_dist_matrix = np.array(features[1]), features[2], features[3]
    cpd_size = cpd_atom_features.shape[0]
    if cpd_size > cpd_cut_len:
        cpd_atom_features = cpd_atom_features[:cpd_cut_len, :]
        cpd_adj_matrix = cpd_adj_matrix[:cpd_cut_len, :cpd_cut_len]
        cpd_dist_matrix = cpd_dist_matrix[:cpd_cut_len, :cpd_cut_len]
    elif cpd_size < cpd_cut_len:
        cpd_atom_features = pad_array(cpd_atom_features, (cpd_cut_len, cpd_atom_features.shape[1]))
        cpd_adj_matrix = pad_array(cpd_adj_matrix, (cpd_cut_len, cpd_cut_len))
        cpd_dist_matrix = pad_array(cpd_dist_matrix, (cpd_cut_len, cpd_cut_len))
    cpd_atom_features = normalize(cpd_atom_features)
    cpd_adj_matrix = normalize_adj(np.array(cpd_adj_matrix))
    return cpd_atom_features, cpd_adj_matrix, cpd_dist_matrix