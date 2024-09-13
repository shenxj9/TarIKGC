from torch.utils.data import Dataset
import numpy as np
import torch


class TrainDataset(Dataset):
    def __init__(self, triplets, num_ent, params, ):
        super(TrainDataset, self).__init__()
        self.p = params
        self.triplets = triplets
        self.num_ent = num_ent

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, item):
        ele = self.triplets[item]
        triple, label = torch.tensor(ele['triple'], dtype=torch.long), np.int32(ele['label'])
        label = self.get_label(label)
        return triple, label

    def get_label(self, label):
        y = np.zeros([self.num_ent], dtype=np.float32)
        y[label] = 1
        return torch.tensor(y, dtype=torch.float32)


class TestDataset(Dataset):
    def __init__(self, triplets, num_ent,params):
        super(TestDataset, self).__init__()
        self.p = params
        self.triplets = triplets
        self.num_ent = num_ent

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, item):
        ele = self.triplets[item]
        triple, label = torch.tensor(ele['triple'], dtype=torch.long), np.int32(ele['label'])
        label = self.get_label(label)
        return triple, label

    def get_label(self, label):

        y = np.zeros([self.num_ent], dtype=np.float32)
        y[label] = 1
        return torch.tensor(y, dtype=torch.float32)


class TestDataset1(Dataset):
    def __init__(self, triplets, num_ent,params):
        super(TestDataset1, self).__init__()
        self.p = params
        self.triplets = triplets
        self.num_ent = num_ent

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, item):
        ele = self.triplets[item]
        triple, label = torch.tensor(ele['triple'], dtype=torch.long), np.int32(ele['label'])
        label = self.get_label(label)
        return triple, label

    def get_label(self, label):

        y = np.zeros([self.num_ent], dtype=np.float32)
        y[label] = 1
        return torch.tensor(y, dtype=torch.float32)


from torch.utils.data import DataLoader
def get_data_iter(triplets,params, num_ent):
    """
    get data loader for train, valid and Repositioning_all section
    :return: dict
    """

    def get_data_loader(dataset_class, num_ent, split):
        return DataLoader(
            dataset_class(triplets[split], num_ent, params),
            batch_size=params.batch_size,
            shuffle=True,
            num_workers=params.num_workers
        )

    return {
        'train': get_data_loader(TrainDataset, num_ent,'train'),
        'test': get_data_loader(TestDataset, num_ent,'test'),
        'test_neg': get_data_loader(TestDataset1, num_ent,'test_neg'),
    }