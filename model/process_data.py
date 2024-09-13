from collections import defaultdict as ddict

def process(dataset,):

    sr2o_neg = ddict(set)
    for subj, obj in dataset['test_neg']:
        sr2o_neg[subj].add(obj)
    sr2o_test_neg = {k: list(v) for k, v in sr2o_neg.items()}

    sr2o = ddict(set)
    for subj, rel, obj in dataset['train']:
        sr2o[(subj, rel)].add(obj)
    sr2o_train = {k: list(v) for k, v in sr2o.items()}

    for subj, rel, obj in dataset['test']:
        sr2o[(subj, rel)].add(obj)
    sr2o_all = {k: list(v) for k, v in sr2o.items()}

    triplets = ddict(list)

    for subj, obj in sr2o_test_neg.items():
        triplets['test_neg'].append({'triple': (subj, 0), 'label': sr2o_test_neg[subj]})

    for (subj, rel), obj in sr2o_train.items():
        triplets['train'].append({'triple': (subj, rel, -1), 'label': sr2o_train[(subj, rel)]})

    for subj, rel, obj in dataset['test']:
        triplets['test'].append({'triple': (subj, rel, obj), 'label': sr2o_all[(subj, rel)]})

    triplets = dict(triplets)
    return triplets


def to_triplets(triplets, entity_dict, relation_dict):
    l = []
    for triplet in triplets:
        s = entity_dict[triplet[0]]
        r = relation_dict[triplet[1]]
        o = entity_dict[triplet[2]]
        l.append([s, r, o])
    return l

import torch
import dgl
import numpy as np

def build_graph(self):
    g = dgl.graph((self.train_data[:, 0], self.train_data[:, 2]), num_nodes=self.num_ent)
    g.add_edges(self.train_data[:, 2], self.train_data[:, 0])
    g.edata['rel_id'] = torch.from_numpy(np.hstack((self.train_data[:, 1], self.train_data[:, 1])))
    return g

def to_label(x, t):
    return np.array([0 if i < t else 1 for i in x])

def save_model(path,model,parms,optimizer,epoch):
    """
    Function to save a model. It saves the model parameters, best validation scores,
    best epoch corresponding to best validation, state of the optimizer and all arguments for the run.
    :param path: path where the model is saved
    :return:
    """
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'args': vars(parms),
        'saved_epoch': epoch + 1,
    }
    torch.save(state, path)


def build_graph(train_data,num_ent):
    g = dgl.graph((train_data[:, 0], train_data[:, 2]), num_nodes=num_ent)
    g.add_edges(train_data[:, 2], train_data[:, 0])
    g.edata['rel_id'] = torch.from_numpy(np.hstack((train_data[:, 1], train_data[:, 1])))
    return g






