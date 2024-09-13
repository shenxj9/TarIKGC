import pandas as pd
import numpy as np
from collections import defaultdict as ddict
def generated_matrix(num_subj,num_ent,num_drugs,train_data, train_neg, test_neg):
    matrix = np.zeros((num_subj, num_ent), dtype=np.int32)
    for i in range(len(train_data)):
        matrix[train_data[i][0], train_data[i][2]] = 1
    for i in range(len(train_neg)):
        matrix[train_neg[i][0], train_neg[i][1]] = -1
    for i in range(len(test_neg)):
        matrix[test_neg[i][0], test_neg[i][1]] = -2

    drug_target_dict = ddict(list)
    for i in range(num_drugs):
        index = [j for j in range(num_drugs, num_subj) if matrix[i][j] == 1]
        drug_target_dict[i].extend([len(index), index])

    drug_target_neg_dict = ddict(list)
    for i in range(num_drugs):
        index = [j for j in range(num_drugs,num_subj) if matrix[i][j] == -1]
        drug_target_neg_dict[i].extend([len(index), index])

    test_neg_dict = ddict(list)
    for i in range(num_drugs):
        index = [j for j in range(num_drugs,num_subj) if matrix[i][j] == -2]
        test_neg_dict[i].extend([len(index), index])

    drug_disease_dict = ddict(list)
    for i in range(num_drugs):
        index = [j for j in range(num_subj,num_ent) if matrix[i][j] == 1]
        drug_disease_dict[i].extend([len(index), index])

    drug_disease_neg_dict = ddict(list)
    for i in range(num_drugs):
        index = [j for j in range(num_subj,num_ent) if matrix[i][j] == -1]
        drug_disease_neg_dict[i].extend([len(index), index])

    target_disease_dict = ddict(list)
    for i in range(num_drugs,num_subj):
        index = [j for j in range(num_subj,num_ent) if matrix[i][j] == 1]
        target_disease_dict[i].extend([len(index), index])

    target_disease_neg_dict = ddict(list)
    for i in range(num_drugs,num_subj):
        index = [j for j in range(num_subj,num_ent) if matrix[i][j] == -1]
        target_disease_neg_dict[i].extend([len(index), index])

    return drug_target_dict, drug_target_neg_dict, test_neg_dict, drug_disease_dict, drug_disease_neg_dict, target_disease_dict, target_disease_neg_dict

