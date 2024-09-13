import numpy as np
import os
import pandas as pd
import torch
import random
from model.process_data import to_triplets
from model.atom_feature import get_compound_feature
import pickle

def _read_dictionary(filename):
    d = {}
    with open(filename, 'r+') as f:
        for line in f:
            line = line.strip().split('\t')
            d[line[1]] = int(line[0])
    return d

def get_kg_data(p):
    num_drugs = 20552
    num_targets = 775
    num_diseases = 709
    num_subj = num_drugs + num_targets
    num_obj = num_targets + num_diseases
    neg_ratio = p.neg_ratio
    entity_path = os.path.join(f'./dataset', 'entityid.csv')
    relation_path = os.path.join(f'./dataset', 'relations.csv')
    data = os.path.join(f'./dataset', 'kg.csv')
    df = pd.read_csv(data, header=None, sep='\t')
    entity_dict = _read_dictionary(entity_path)
    relation_dict = _read_dictionary(relation_path)
    all_df = pd.read_csv(data, header=None, sep='\t')
    inter_matrix = np.zeros((num_subj, num_obj), dtype=np.int32)
    for i in range(len(all_df)):
        a = entity_dict[all_df.loc[i][0]]
        b = entity_dict[all_df.loc[i][2]]
        inter_matrix[a, b - num_drugs] = 1
    drug_target_negs,  disease_negs, drug_target_data, disease_related_data\
        = get_negs(df, num_drugs, num_targets, num_diseases, num_subj, entity_dict, relation_dict, neg_ratio)
    nums_nodes_dict = {'num_drugs':num_drugs,'num_targets':num_targets,'num_diseases':num_diseases, 'num_subj':num_subj,"num_obj":num_obj,}
    return drug_target_negs, disease_negs, drug_target_data, disease_related_data, entity_dict, relation_dict, nums_nodes_dict

def feature_base(df, entity_dict):
    file = rf'./dataset/mol_feature.pt'
    if os.path.exists(file):
        mol_feature = torch.load(file)
    else:
        mol_feature = {}
        for i in range(len(df)):
            id = entity_dict[df.iloc[i].id]
            cpd_atom_features, cpd_adj_matrix, cpd_dist_matrix = get_compound_feature(df.loc[i]['smiles'])
            mol_feature[id] = [cpd_atom_features, cpd_adj_matrix, cpd_dist_matrix]
        torch.save(mol_feature,file)
    return mol_feature

def get_mol_feature(path, entity_dict):
    file_path = os.path.join(f'./dataset', path)
    df = pd.read_csv(file_path, sep='\t', header=None, names=['id', 'smiles'])
    smi_dict = {}
    for i in range(len(df)):
        id = entity_dict[df.iloc[i].id]
        smi_dict[id] = df.loc[i]['smiles']
    save_path = rf'./dataset/smi_feature.pt'
    torch.save(smi_dict, save_path)
    mol_feature = feature_base(df,entity_dict)
    smi_dict = sorted(smi_dict.items(), key=lambda x: x[0])
    return smi_dict, mol_feature

def get_negs(df, num_drugs, num_targets, num_diseases, num_subj, entity_dict, relation_dict, neg_ratio):
    df_drug_target = df[df[1] == 'drug_target']
    df_drug_disease = df[df[1] == 'drug_disease']
    df_target_disease = df[df[1] == 'target_disease']
    df_disease = df[df[1] != 'drug_target']
    drug_target_negs = get_negs_sub(num_drugs,num_targets,df_drug_target, 0, num_drugs, entity_dict, neg_ratio)
    drug_disease_negs = get_negs_sub(num_drugs, num_diseases, df_drug_disease, 0, num_subj, entity_dict, neg_ratio)
    target_disease_negs = get_negs_sub(num_targets, num_diseases,df_target_disease, num_drugs,num_subj, entity_dict, neg_ratio)
    negs = []
    negs.extend(drug_target_negs)
    negs.extend(drug_disease_negs)
    negs.extend(target_disease_negs)
    disease_negs = []
    disease_negs.extend(drug_disease_negs)
    disease_negs.extend(target_disease_negs)
    drug_target_data = np.asarray(to_triplets(df_drug_target.values, entity_dict,relation_dict))
    disease_related_data = np.asarray(to_triplets(df_disease.values, entity_dict,relation_dict))
    return drug_target_negs, disease_negs,drug_target_data, disease_related_data

def get_negs_sub(head, tail, df, n1, n2, entity_dict, neg_ratio):
    df = df.reset_index()
    matrix = np.zeros((head, tail), dtype=np.int32)
    for i in range(len(df)):
        a = entity_dict[df.loc[i][0]]
        b = entity_dict[df.loc[i][2]]
        matrix[a - n1, b - n2] = 1

    train_negs = dict()
    for i in range(head):
        neg_index = [j for j in range(tail) if matrix[i][j] == 0]
        num = int(sum(matrix[i]).item()) * neg_ratio
        if len(neg_index) >= num:
            train_index = random.sample(neg_index, num)
        else:
            train_index = neg_index
        train_negs[i] = train_index

    negs = []
    for i in range(head):
        for j in train_negs[i]:
            negs.append([i + n1, j + n2])
    return negs

def get_protein_feaure():
    f = r'./dataset/uniprot_go.csv'
    df = pd.read_csv(f, sep = '\t')
    df = df.fillna('-')
    f1 = r'./dataset/go_dict.csv'
    go_terms = pd.read_csv(f1, sep = '\t', )['ID'].tolist()
    go_dict = dict(zip(go_terms, [i for i in range(len(go_terms))]))
    features_length = len(go_dict)
    file = rf'./dataset/Go_data.pt'
    if os.path.exists(file):
        data = torch.load(file)
    else:
        data = torch.zeros((len(df), features_length), dtype=torch.float32)
        for i, row in enumerate(df.itertuples()):
            if row.Gene_Ontology_IDs != '-':
                Go_IDS = row.Gene_Ontology_IDs.split(';')
                for Go in Go_IDS:
                    data[i, go_dict[Go]] = 1
        torch.save(data, file)
    return  data,features_length

def get_disease_feature():

    def load_obj(name):
        try:
            f = open(name + '.pkl', 'rb')
        except IOError:
            return None
        else:
            return pickle.load(f)

    f1 = f"./dataset/icd_code_vec_GatorTron_OG_finetuning_py37"
    icd_code_vec = load_obj(f1)
    f2 = './dataset/Disease_dict.csv'
    df = pd.read_csv(f2, sep='\t', header=None)
    Diseases_list = df[1].unique().tolist()
    emb = []
    for Diseases in Diseases_list:
        emb.append(icd_code_vec[Diseases])
    emb = np.concatenate(emb).astype(np.float32)
    return torch.from_numpy(emb)
