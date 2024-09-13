import pandas as pd
import pickle
import torch
import numpy as np
def load_obj(name):
    try:
        f = open(name + '.pkl', 'rb')
    except IOError:
        return None
    else:
        return pickle.load(f)

icd_vec = load_obj("./dataset/icd_code_vec_GatorTron-OG_finetuning_20230324")
print(len(icd_vec))

f = './datasets/Disease_dict.csv'
df = pd.read_csv(f,sep = '\t', header = None)
Diseases_list = df[1].unique().tolist()
emb = []
for Diseases in Diseases_list:
    emb.append(icd_vec[Diseases])
emb = np.concatenate(emb)

f = f"./datasets/KG_10/icd_code_vec_GatorTron_OG_finetuning.pkl"
with open (f, 'wb') as file:
    pickle.dump(torch.from_numpy(emb), file)




