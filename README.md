#  TarIKGC: a target identification tool based on semantics enhanced knowledge graph completion 


- This repository contains the source code ,the data and trained models.


## Train
Model training can be started by running the `main.py` script:
```bash
python main.py  --gpu 2  --epoch 40
```


**Notes**:

Pre-download and Pre-generate Necessary Files to Save Model Training Time:

1. Download the `disease embedding` file from this [link](https://github.com/YeongChanLee/ICD2Vec/blob/main/model/GatorTron-OG_icd2vec_finetuning/icd_code_vec_GatorTron-OG_finetuning_24354codes.zip). Move the downloaded file to the `dataset/` directory and then execute the script `ICD_embedding.py`.

2. Run the script `preprocess.py` to generate the `mol_feature.pt` file.





Training the model will display folder `output/` with the following structure:

```
KG
└── |"{now_time}_max{self.p.max_epochs}
        ├── models
        ├── results
```





- You can run the `reposition_lab.py` script to predict the targets of a molecule of interest. you will obtain a target recommendation list. For example, You can reproduce the prediction results of the CDK2 inhibitor discovery by running the following scripts:

```bash
python reposition_lab.py --gpu 0 
```



