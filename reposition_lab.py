import argparse
import logging
import numpy as np
import random
from pathlib import Path
import torch
from model.interaction import generated_matrix
from model.tarikgc import TarIKGC
from model.process_data  import process, build_graph
from model.data_set  import get_data_iter
import os
import pandas as pd
from torch.optim.lr_scheduler import StepLR
logging.getLogger().setLevel(logging.INFO)
from model.semantics import get_kg_data, get_mol_feature, get_protein_feaure, get_disease_feature


def load_predict(entity_dict, relation_dict):
    l = []
    compounds = []
    filename = f'./dataset/mol_info.csv'
    df = pd.read_csv(filename, sep = '\t', header = None)
    for i in range(len(df)):
        compound = df.iloc[i][0]
        l.append([entity_dict[compound], relation_dict['drug_target']])
        compounds.append(compound)
    return torch.tensor(l).to("cuda"), compounds


class main(object):
    def __init__(self, params):
        self.p = params
        self.prj_path = Path(__file__).parent.resolve()
        self.init_data()
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def init_data(self):
        self.drug_target_negs, self.disease_negs, self.drug_target_data, self.disease_related_data, self.entity_dict, self.relation_dict, nums_dict = get_kg_data(
            self.p)
        self.num_ent = len(self.entity_dict)
        self.num_rels = len(self.relation_dict)
        self.num_subj, self.num_obj = nums_dict['num_subj'], nums_dict['num_obj']
        self.p.num_drugs, self.p.num_targets, self.p.num_diseases = nums_dict['num_drugs'], nums_dict['num_targets'], \
                                                                    nums_dict['num_diseases']
        self.train_data, self.test_data = self.drug_target_data, self.drug_target_data[[0]]
        self.train_neg, self.test_neg = self.drug_target_negs, self.drug_target_negs[:2]
        self.train_data = np.concatenate((self.train_data, self.disease_related_data), axis=0)
        self.train_neg = np.concatenate((self.train_neg, self.disease_negs), axis=0)
        self.smi_dict, self.mol_feature = get_mol_feature(self.p.mol_path, self.entity_dict)
        self.protein_go, self.features_length  = get_protein_feaure()
        self.disease_feature = get_disease_feature()
        self.drug_target_dict, self.drug_target_neg_dict, self.test_neg_dict, self.drug_disease_dict, self.drug_disease_neg_dict, self.target_disease_dict, self.target_disease_neg_dict = \
            generated_matrix(self.num_subj,  self.num_ent, self.p.num_drugs, self.train_data,
                             self.train_neg, self.test_neg)
        self.triplets = process({'train': self.train_data, 'test': self.test_data,'test_neg':self.test_neg})
        self.data_iter = get_data_iter(self.triplets, self.p, self.num_ent)
        self.columns = list(self.entity_dict.keys())[self.p.num_drugs: self.p.num_drugs + self.p.num_targets]

    def predict(self, listpath):
        if self.p.gpu >= 0:
            self.g = build_graph(self.train_data, self.num_ent).to("cuda")
        else:
            self.g = build_graph(self.train_data)
        self.model= self.get_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.p.lr)
        self.scheduler = StepLR(self.optimizer, step_size=2, gamma=0.98)
        self.predicted, compounds = load_predict(self.entity_dict, self.relation_dict)
        preds_list = []

        for path in listpath:
            self.load_model(path)
            self.model.eval()
            with torch.no_grad():
                subj = self.predicted[:, 0]
                rel = self.predicted[:, 1]
                _,preds= self.model(self.g, subj, rel)
                preds = preds[:,:self.p.num_targets]
                # ranks = 1 + torch.argsort(torch.argsort(preds, dim=1, descending=True), dim=1, descending=False)
                # print(preds[:,self.entity_dict['P24941']-self.p.num_drugs],ranks[:, self.entity_dict['P24941']-self.p.num_drugs])
                preds_list.append(preds)
        preds = torch.stack(preds_list, dim=0).mean(dim=0)
        ranks = 1 + torch.argsort(torch.argsort(preds, dim=1, descending=True), dim=1, descending=False)
        result_ranks = ranks.cpu().numpy()
        result_scores = preds.cpu().numpy()

        df_scores = pd.DataFrame(result_scores)
        df_scores.columns = self.columns
        df_scores.index = compounds

        df_ranks = pd.DataFrame(result_ranks)
        df_ranks.columns = self.columns
        df_ranks.index = compounds

        ranks = df_ranks['P24941'].tolist()
        scores = df_scores['P24941'].tolist()
        df_result = pd.DataFrame({'rank':ranks ,'score':scores})
        df_result.index = compounds

        f = rf'./output/score.csv'
        df_scores.to_csv(f)
        f = rf'./output/rank.csv'
        df_ranks.to_csv(f)

            # print(df_ranks.values.shape, ranks)
            # count = len([i for i in ranks if i < 50])
            # if count == 2:
            #     print('get:_________________________', )
            #     b = path.split('/')[2]
            #     c = path.split('/')[4].split('.')[0]
            #     f = rf'./output/web/predict_{b}_{c}.csv'
            #     df_result['P24941'].to_csv(f)


    def load_model(self, path,):
        """
        Function to load a saved model
        :param path: path where model is loaded
        :return:
        """
        state = torch.load(path)
        self.model.load_state_dict(state['model'],False)
        self.optimizer.load_state_dict(state['optimizer'])
        self.saved_epoch = state['saved_epoch']

    def get_model(self):
        model = TarIKGC(num_ent=self.num_ent, num_rel=self.num_rels, num_base=self.p.num_bases,
                             init_dim=self.p.init_dim,  embed_dim=self.p.embed_dim,
                             smi_dict =self.smi_dict, params=self.p,mol_feature = self.mol_feature,
                            protein_go = self.protein_go, features_length = self.features_length,
                            disease_feature = self.disease_feature)

        if self.p.gpu >= 0:
            model.to("cuda")
        return model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parser For Arguments',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mol_path', dest='mol_path',default='drug_dict.csv')
    parser.add_argument('--num_drugs', dest='num_drugs', default=None)
    parser.add_argument('--num_diseases', dest='num_diseases', default=None)
    parser.add_argument('--num_targets', dest='num_targets', default=None)
    parser.add_argument('--nfold', dest='nfold', default=10,type=int,
                        help='Dataset to use,')
    parser.add_argument('--neg_ratio', dest='neg_ratio', default=1,type=int,
                        help='Ratio of positive and negative interactions')
    parser.add_argument('--batch', dest='batch_size',
                        default=512, type=int, help='Batch size')
    parser.add_argument('--gpu', type=int, default=4)
    parser.add_argument('--epoch', dest='max_epochs',
                        type=int, default=500, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.003,
                        help='Starting Learning Rate')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of processes to construct batches')
    parser.add_argument('--seed', dest='seed', default=1234,
                        type=int, help='Seed for randomization')
    parser.add_argument('--restore', dest='restore', action='store_true',
                        help='Restore from the previously saved model')
    parser.add_argument('--model_saved_path', dest='model_saved_path',
                        help='Path of the previously saved model')
    parser.add_argument('--bias', dest='bias', action='store_true',default=True,
                        help='Whether to use bias in the model')
    parser.add_argument('--init_dim', dest='init_dim', default=128, type=int,
                        help='Initial dimension size for entities and relations')
    parser.add_argument('--embed_dim', dest='embed_dim', default=128, type=int,
                        help='Embedding dimension to give as input to score function')
    parser.add_argument('--num_bases', dest='num_bases', default=None, type=int,
                        help='Number of basis relation vectors to use')
    parser.add_argument('--ent_drop', dest='ent_drop', default=0.3, type=float)
    parser.add_argument('--rel_drop', dest='rel_drop', default=0.3, type=float)
    args = parser.parse_args()
    model_paths = [
        r'save_model/Ensemble1.pt',
        r'save_model/Ensemble2.pt',
        r'save_model/Ensemble3.pt',
        r'save_model/Ensemble4.pt',
        r'save_model/Ensemble5.pt',
    ]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    result= main(args)
    result.predict(model_paths)





