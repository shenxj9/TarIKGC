import os
import argparse
import logging
from numpy import arange
from numpy import argmax
import numpy as np
import random
from pathlib import Path
import torch
from model.interaction import generated_matrix
from model.tarikgc import TarIKGC
from model.process_data  import process, save_model, to_label, build_graph
from model.data_set  import get_data_iter
from datetime import datetime
import pandas as pd
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score,  recall_score, accuracy_score,precision_recall_curve,auc,f1_score
logging.getLogger().setLevel(logging.INFO)
from model.semantics import get_kg_data, get_mol_feature, get_protein_feaure, get_disease_feature

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
        drug_target_negs, disease_negs, drug_target_data, disease_related_data,entity_dict,relation_dict,nums_dict = get_kg_data(self.p)
        self.num_ent = len(entity_dict)
        self.num_rels = len(relation_dict)
        self.num_subj,self.num_obj =nums_dict['num_subj'], nums_dict['num_obj']
        self.p.num_drugs, self.p.num_targets, self.p.num_diseases= nums_dict['num_drugs'], nums_dict['num_targets'], nums_dict['num_diseases']
        self.drug_target_data,self.disease_related_data = np.array(drug_target_data), np.array(disease_related_data)
        self.drug_target_negs, self.disease_negs = np.array(drug_target_negs),disease_negs
        self.smi_dict, self.mol_feature = get_mol_feature(self.p.mol_path, entity_dict)
        self.protein_go, self.features_length  = get_protein_feaure()
        self.disease_feature = get_disease_feature()
        self.models_path, self.result_path = self.mkdir_root()

    def load_model(self, path,):
        state = torch.load(path)
        self.model.load_state_dict(state['model'],False)
        self.optimizer.load_state_dict(state['optimizer'])
        self.saved_epoch = state['saved_epoch']


    def mkdir_root(self):
        now_time = datetime.now().strftime('%y_%m_%d__%H_%M')
        root = self.prj_path /'output'
        if not root.exists():
            root.mkdir()
        save_root = root / f"{now_time}_max{self.p.max_epochs}"
        models_path = save_root/'models'
        result_path = save_root /'results'
        self.result_path = result_path
        if not save_root.exists():
            save_root.mkdir()
        if not models_path.exists():
            models_path.mkdir()
        if not result_path.exists():
            result_path.mkdir()
        return models_path, result_path

    def cross_validation(self):
        val_results_list = []
        kf = KFold(n_splits=self.p.nfold, shuffle=True, random_state=200)
        nfold = 0
        for (train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx) in zip(kf.split(self.drug_target_data),
                                                                                kf.split(self.drug_target_negs)):
            print(nfold, 'nfold*************************')
            nfold = nfold + 1
            self.train_data, self.test_data = self.drug_target_data[train_pos_idx], self.drug_target_data[test_pos_idx]
            self.train_neg, self.test_neg = self.drug_target_negs[train_neg_idx], self.drug_target_negs[test_neg_idx]
            self.train_data = np.concatenate((self.train_data, self.disease_related_data), axis=0)
            self.train_neg = np.concatenate((self.train_neg, self.disease_negs), axis=0)
            self.drug_target_dict, self.drug_target_neg_dict, self.test_neg_dict, self.drug_disease_dict, self.drug_disease_neg_dict, self.target_disease_dict, self.target_disease_neg_dict = \
                generated_matrix(self.num_subj,  self.num_ent, self.p.num_drugs, self.train_data,
                                 self.train_neg, self.test_neg)
            self.triplets = process({'train': self.train_data, 'test': self.test_data,'test_neg':self.test_neg})
            self.data_iter = get_data_iter(self.triplets, self.p, self.num_ent)
            if self.p.gpu >= 0:
                self.g = build_graph(self.train_data, self.num_ent).to("cuda")
            else:
                self.g = build_graph(self.train_data)

            self.model = TarIKGC(num_ent=self.num_ent, num_rel=self.num_rels, num_base=self.p.num_bases,
                            init_dim=self.p.init_dim, embed_dim=self.p.embed_dim,
                            smi_dict=self.smi_dict, params=self.p, mol_feature=self.mol_feature,
                            protein_go=self.protein_go, features_length=self.features_length,
                            disease_feature=self.disease_feature)
            if self.p.gpu >= 0:
                self.model.to("cuda")
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.p.lr)
            self.scheduler = StepLR(self.optimizer, step_size=2, gamma=0.98)
            if self.p.restore:
                self.load_model(self.p.model_saved_path)
            else:
                self.saved_epoch = 0
            for epoch in range(self.saved_epoch,self.saved_epoch+self.p.max_epochs):
                print('Time:',datetime.now().strftime('%H:%M'),'epoch:',epoch)
                train_loss = self.train()
                self.scheduler.step()
                val_results, all_pred, labels, pred = self.predict()
                self.logger.info(
                    f"[Epoch {epoch+1}]"
                    f"Training Loss: {train_loss:.2}, MR: {val_results['mr']:.4},MRR: {val_results['mrr']:.3},"
                    f"@10:{val_results['hits@10']:.2},'roc_auc':{val_results['roc_auc']:.2},'pr_auc':{val_results['pr_auc']:.2},"
                    f"'recall':{val_results['recall']:.2},")

                val_results['nfold'] = nfold
                val_results['epoch'] = epoch + 1
                val_results = pd.DataFrame(val_results, index=[0])
                val_results_list.append(val_results)

                val_results_output = pd.concat(val_results_list)
                save_path = self.result_path / f'valid_result_{nfold}_{epoch}.csv'
                val_results_output.to_csv(save_path, sep='\t', index=False)
        val_results_output = pd.concat(val_results_list)
        save_path = self.result_path/f'valid_result_{nfold}_{epoch}.csv'
        val_results_output.to_csv(save_path, sep='\t',index=False)

    def train(self):
        self.model.train()
        losses = []
        train_iter = self.data_iter['train']
        for step, (triplets, labels) in enumerate(train_iter):
            if self.p.gpu >= 0:
                triplets= triplets.to("cuda")
            subj, rel = triplets[:, 0], triplets[:, 1]
            train_pos = []
            train_neg = []
            for m, n in enumerate(subj.tolist()):
                if self.drug_target_dict[n] != []:
                    train_pos.extend([[m, j] for j in self.drug_target_dict[n][1]])
                if self.drug_target_neg_dict[n] != []:
                    train_neg.extend([[m, j] for j in self.drug_target_neg_dict[n][1]])
            index_i = torch.hstack((torch.tensor(train_pos)[:, 0], torch.tensor(train_neg)[:, 0]))
            index_j = torch.hstack((torch.tensor(train_pos)[:, 1], torch.tensor(train_neg)[:, 1]))
            pred, _= self.model(self.g, subj, rel)
            target_pred0 = pred[index_i, index_j - self.p.num_drugs]
            labels_true = torch.hstack((torch.ones(len(train_pos)), torch.zeros(len(train_neg))))
            loss = self.model.calc_loss(target_pred0, labels_true.to("cuda"))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        losses = [round(i, 5) for i in losses]
        loss = np.mean(losses)
        return loss

    def predict(self):
        self.model.eval()
        with torch.no_grad():
            results = dict()
            test_iter = self.data_iter['test']
            pos_preds = []
            rank_pred = []
            for step, (triplets, labels) in enumerate(test_iter):
                triplets,labels = triplets.to("cuda"),labels.to("cuda")
                subj, rel, obj = triplets[:, 0], triplets[:, 1], triplets[:, 2]-self.p.num_drugs
                _, pred = self.model(self.g, subj, rel)
                pred = pred[:,:self.p.num_targets]
                b_range = torch.arange(pred.shape[0], device="cuda")
                target_pred = pred[b_range, obj]
                pred = torch.where(
                    labels[:,self.p.num_drugs:self.num_subj].bool(), -torch.ones_like(pred) * 10000000, pred)
                pred[b_range, obj] = target_pred
                ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]
                pos_preds.extend(target_pred.tolist())
                rank_pred.extend(ranks.tolist())

            test_iter = self.data_iter['test_neg']
            neg_preds = []
            for step, (triplets, labels) in enumerate(test_iter):
                triplets,labels = triplets.to("cuda"),labels.to("cuda")
                subj, rel = triplets[:, 0], triplets[:, 1]
                _, pred = self.model(self.g, subj, rel)
                pred = pred[:, :self.p.num_targets]
                target_pred = pred[labels[:,self.p.num_drugs:self.num_subj].bool()]
                neg_preds.extend(target_pred.tolist())

            subj, rel = triplets[:, 0], triplets[:, 1]
            _, pred = self.model(self.g, subj, rel)
            pred = pred[:, :self.p.num_targets]
            target_pred = pred[labels[:, self.p.num_drugs:self.num_subj].bool()]
            neg_preds.extend(target_pred.tolist())

            all_pred = torch.tensor(pos_preds+neg_preds)
            labels = torch.hstack((torch.ones(len(pos_preds)), torch.zeros(len(neg_preds))))
            results['roc_auc'] = roc_auc_score(labels, all_pred)
            precision, recall, thresholds = precision_recall_curve(labels, all_pred, pos_label=1)
            results['pr_auc'] = auc(recall, precision)
            thresholds = arange(0, 0.3, 0.01)
            scores = [f1_score(labels, to_label(all_pred, t)) for t in thresholds]
            ix = argmax(scores)
            best_threshold = thresholds[ix]
            target_pred_labels = np.array([0 if i < best_threshold else 1 for i in all_pred])
            results['accuracy'] = accuracy_score(labels, target_pred_labels)
            results['recall'] = recall_score(labels, target_pred_labels)
            ranks = torch.tensor([rank_pred])
            count = torch.numel(ranks)
            ranks = ranks.float()
            results['mr'] = round(torch.sum(ranks).item() / count, 5)
            results['mrr'] = round(torch.sum(1.0 / ranks).item() / count, 5)
            for k in [1, 3, 10]:
                results[f'hits@{k}'] = round(torch.numel(
                    ranks[ranks <= k])/ count, 5)
        return results, all_pred, labels, pred



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
    parser.add_argument('--seed', dest='seed', default=123,
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
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    result= main(args)
    result.cross_validation()
