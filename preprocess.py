
import os
import pandas as pd
import torch
from model.atom_feature import get_compound_feature



def feature_base(df, entity_dict):
    file = rf'./dataset/mol_feature.pt'
    if os.path.exists(file):
        mol_feature = torch.load(file)
    else:
        mol_feature = {}
        for i in range(len(df)):
            id = entity_dict[df.iloc[i].id]
            print(id)
            cpd_atom_features, cpd_adj_matrix, cpd_dist_matrix = get_compound_feature(df.loc[i]['smiles'])
            mol_feature[id] = [cpd_atom_features, cpd_adj_matrix, cpd_dist_matrix]
        torch.save(mol_feature,file)


def get_mol_feature(path, entity_dict):
    file_path = os.path.join(f'./dataset', path)
    df = pd.read_csv(file_path, sep='\t', header=None, names=['id', 'smiles'])
    feature_base(df,entity_dict)


def _read_dictionary(filename):
    d = {}
    with open(filename, 'r+') as f:
        for line in f:
            line = line.strip().split('\t')
            d[line[1]] = int(line[0])
    return d

entity_path = os.path.join(f'./dataset', 'entityid.csv')
entity_dict = _read_dictionary(entity_path)
mol_path = r'drug_dict.csv'
get_mol_feature(mol_path, entity_dict)