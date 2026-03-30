import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np


class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, config, training_mode, target_dataset_size=64, subset=False):
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode
        X_train = dataset["samples"]
        y_train = dataset["labels"]
        # shuffle
        data = list(zip(X_train, y_train))
        # np.random.shuffle(data)
        X_train, y_train = zip(*data)
        try:
            X_train, y_train = torch.stack(list(X_train), dim=0), torch.stack(list(y_train), dim=0)
        except:
            X_train, y_train = list(X_train), list(y_train)
            for i in range(len(X_train)):
                X_train[i] = torch.tensor(X_train[i], dtype=torch.float32)
            for i in range(len(y_train)):
                y_train[i] = torch.tensor(y_train[i], dtype=torch.float32)
            X_train, y_train = torch.stack(list(X_train), dim=0), torch.stack(list(y_train), dim=0)
            
        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

        """Align the TS length between source and target datasets"""
        #X_train = X_train[:, :1, :int(config.TSlength_aligned)] # take the first 178 samples

        X_train = X_train[:, :1, :int(config.TSlength_aligned)]

        """Subset for debugging"""
        if subset == True:
            subset_size = target_dataset_size *10
            """if the dimension is larger than 178, take the first 178 dimensions. If multiple channels, take the first channel"""
            X_train = X_train[:subset_size] 
            y_train = y_train[:subset_size]

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        self.len = X_train.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def data_generator(targetdata_path, configs, training_mode, subset = True):

    train_dataset_sleep = torch.load('dataset/SleepEEG/train.pt')
    train_dataset_epil = torch.load('dataset/Epilepsy/train.pt')
    train_dataset_HAR = torch.load('dataset/HAR/train.pt')
    train_dataset_ECG = torch.load('dataset/ECG/train.pt')
    train_dataset_FDA = torch.load('dataset/FD-A/train.pt')
    
    finetune_dataset = torch.load(os.path.join(targetdata_path, "train.pt"))
    test_dataset = torch.load(os.path.join(targetdata_path, "test.pt"))


    """ Dataset notes:
    Epilepsy: train_dataset['samples'].shape = torch.Size([7360, 1, 178]); binary labels [7360] 
    valid: [1840, 1, 178]
    test: [2300, 1, 178]. In test set, 1835 are positive sampels, the positive rate is 0.7978"""
    """sleepEDF: finetune_dataset['samples']: [7786, 1, 3000]"""

    # subset = True # if true, use a subset for debugging.
    train_dataset_sleep = Load_Dataset(train_dataset_sleep, configs[0], training_mode, target_dataset_size=configs[0].batch_size, subset=subset) # for self-supervised, the data are augmented here
    train_dataset_epil = Load_Dataset(train_dataset_epil, configs[1], training_mode, target_dataset_size=configs[1].batch_size, subset=subset) # for self-supervised, the data are augmented here
    train_dataset_HAR = Load_Dataset(train_dataset_HAR, configs[2], training_mode, target_dataset_size=configs[2].batch_size, subset=subset) # for self-supervised, the data are augmented here
    train_dataset_ECG = Load_Dataset(train_dataset_ECG, configs[3], training_mode, target_dataset_size=configs[3].batch_size, subset=subset) # for self-supervised, the data are augmented here
    train_dataset_FDA = Load_Dataset(train_dataset_FDA, configs[4], training_mode, target_dataset_size=configs[4].batch_size, subset=subset) # for self-supervised, the data are augmented here
    

    finetune_dataset = Load_Dataset(finetune_dataset, configs[5], training_mode, target_dataset_size=configs[0].target_batch_size, subset=subset)  # 函数里有shuffle
    if test_dataset['labels'].shape[0]>10*configs[5].target_batch_size:
        test_dataset = Load_Dataset(test_dataset, configs[5], training_mode, target_dataset_size=configs[0].target_batch_size*10, subset=subset)
    else:
        test_dataset = Load_Dataset(test_dataset, configs[5], training_mode, target_dataset_size=configs[0].target_batch_size, subset=subset)

    train_loader_sleep = torch.utils.data.DataLoader(dataset=train_dataset_sleep, batch_size=configs[0].batch_size,
                                               shuffle=True, drop_last=configs[0].drop_last,
                                               num_workers=0)

    train_loader_epil = torch.utils.data.DataLoader(dataset=train_dataset_epil, batch_size=configs[1].batch_size,
                                               shuffle=True, drop_last=configs[1].drop_last,
                                               num_workers=0)
    train_loader_HAR = torch.utils.data.DataLoader(dataset=train_dataset_HAR, batch_size=configs[2].batch_size,
                                               shuffle=True, drop_last=configs[2].drop_last,
                                               num_workers=0)    
    train_loader_ECG = torch.utils.data.DataLoader(dataset=train_dataset_ECG, batch_size=configs[3].batch_size,
                                               shuffle=True, drop_last=configs[3].drop_last,
                                               num_workers=0)
    train_loader_FDA = torch.utils.data.DataLoader(dataset=train_dataset_FDA, batch_size=configs[4].batch_size,
                                               shuffle=True, drop_last=configs[4].drop_last,
                                               num_workers=0)
    

    """ICH_TMT."""  
    
    valid_loader = torch.utils.data.DataLoader(dataset=finetune_dataset, batch_size=configs[5].target_batch_size,  
                                               shuffle=True, drop_last=configs[5].drop_last,
                                               num_workers=0)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs[5].target_batch_size,
                                              shuffle=True, drop_last=False,
                                              num_workers=0)

    train_loader = [train_loader_sleep, train_loader_epil, train_loader_HAR, train_loader_ECG, train_loader_FDA]
    return  train_loader, valid_loader, test_loader

