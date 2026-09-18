import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

class MSL_data_loader(Dataset):
    def __init__(self, root_path, seq_len, pred_len, step=1, mode='train'):
        self.mode = mode
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.step = step
        self.scaler = StandardScaler()

        train_data = np.load(os.path.join(root_path, 'MSL_train.npy'))
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)

        test_data = np.load(os.path.join(root_path, 'MSL_test.npy'))
        test_data = self.scaler.transform(test_data)
        test_label = np.load(os.path.join(root_path, 'MSL_test_label.npy'))

        self.train_data = train_data
        self.test_data = test_data
        self.val_data = test_data
        self.test_label = test_label

    def __len__(self):
        if self.mode == 'train':
            data = self.train_data
        elif self.mode == 'val':
            data = self.val_data
        elif self.mode == 'test':
            data = self.test_data
        else:
            data = self.test_data
        
        if len(data) < self.seq_len + self.pred_len:
            return 0
        return (len(data) - self.seq_len - self.pred_len) // self.step + 1
    
    def __getitem__(self, index):
        idx = index * self.step
        if self.mode == 'train':
            data = self.train_data
            X = np.float32(data[idx: idx + self.seq_len])
            y = np.float32(data[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            return X, y
        elif self.mode == 'val':
            data = self.val_data
            X = np.float32(data[idx: idx + self.seq_len])
            y = np.float32(data[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            return X, y
        elif self.mode == 'test':
            data =  self.test_data
            X = np.float32(data[idx: idx + self.seq_len])
            y = np.float32(data[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            # label = np.float32(self.test_label[idx: idx + self.seq_len + self.pred_len])
            label = np.float32(self.test_label[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            return X, y, label
        else:
            data =  self.test_data
            X = np.float32(data[idx: idx + self.seq_len])
            y = np.float32(data[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            label = np.float32(self.test_label[idx + self.seq_len: idx + self.seq_len + self.pred_len])
        return X, y, label

class PSM_data_loader(Dataset):
    def __init__(self, root_path, seq_len, pred_len, step=1, mode='train'):
        self.mode = mode
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.step = step
        self.scaler = StandardScaler()

        train_df = pd.read_csv(os.path.join(root_path, 'train.csv'))
        train_data = train_df.values[:, 1:]
        train_data = np.nan_to_num(train_data)
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)

        test_df = pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_df.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        test_data = self.scaler.transform(test_data)

        test_label = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]

        self.train_data = train_data
        self.test_data = test_data
        self.val_data = test_data
        self.test_label = test_label
    
    def __len__(self):
        if self.mode == 'train':
            data = self.train_data
        elif self.mode == 'val':
            data = self.val_data
        elif self.mode == 'test':
            data = self.test_data
        else:
            data = self.test_data

        if len(data) < self.seq_len + self.pred_len:
            return 0
        return (len(data) - self.seq_len - self.pred_len) // self.step + 1

    def __getitem__(self, index):
        idx = index * self.step
        if self.mode == 'train':
            data = self.train_data
            X = np.float32(data[idx: idx + self.seq_len])
            y = np.float32(data[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            return X, y
        elif self.mode == 'val':
            data = self.test_data
            X = np.float32(data[idx: idx + self.seq_len])
            y = np.float32(data[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            return X, y
        elif self.mode == 'test':
            data = self.test_data
            X = np.float32(data[idx: idx + self.seq_len])
            y = np.float32(data[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            label = np.float32(self.test_label[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            return X, y, label
        else:
            data = self.test_data
            X = np.float32(data[idx: idx + self.seq_len])
            y = np.float32(data[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            label = np.float32(self.test_label[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            return X, y, label
        
class SMAP_data_loader(Dataset):
    def __init__(self, root_path, seq_len, pred_len, step=1, mode='train'):
        self.mode = mode
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.step = step
        self.scaler = StandardScaler()

        train_data = np.load(os.path.join(root_path, 'SMAP_train.npy'))
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)

        test_data = np.load(os.path.join(root_path, 'SMAP_test.npy'))
        test_data = self.scaler.transform(test_data)
        test_label = np.load(os.path.join(root_path, 'SMAP_test_label.npy'))

        self.train_data = train_data
        self.test_data = test_data
        self.val_data = test_data
        self.test_label = test_label

    def __len__(self):
        if self.mode == 'train':
            data = self.train_data
        elif self.mode == 'val':
            data = self.val_data
        elif self.mode == 'test':
            data = self.test_data
        else:
            data = self.test_data
        if len(data) < self.seq_len + self.pred_len:
            return 0
        return (len(data) - self.seq_len - self.pred_len) // self.step + 1

    def __getitem__(self, index):
        idx = index * self.step
        if self.mode == 'train':
            data = self.train_data
            X = np.float32(data[idx: idx + self.seq_len])
            y = np.float32(data[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            return X, y
        elif self.mode == 'val':
            data = self.test_data
            X = np.float32(data[idx: idx + self.seq_len])
            y = np.float32(data[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            return X, y
        elif self.mode == 'test':
            data = self.test_data
            X = np.float32(data[idx: idx + self.seq_len])
            y = np.float32(data[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            label = np.float32(self.test_label[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            return X, y, label
        else:
            data = self.test_data
            X = np.float32(data[idx: idx + self.seq_len])
            y = np.float32(data[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            label = np.float32(self.test_label[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            return X, y, label

class SMD_data_loader(Dataset):
    def __init__(self, root_path, seq_len, pred_len, step=1, mode='train'):
        self.mode = mode
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.step = step
        self.scaler = StandardScaler()

        train_data = np.load(os.path.join(root_path, 'SMD_train.npy'))[:, :]
        data_len = len(train_data)
        train_data = train_data[int(data_len * 0.3):, :]
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)

        test_data = np.load(os.path.join(root_path, 'SMD_test.npy'))[:, :]
        test_data = self.scaler.transform(test_data)
        test_label = np.load(os.path.join(root_path, 'SMD_test_label.npy'))

        self.train_data = train_data
        self.test_data = test_data
        self.val_data = test_data
        self.test_label = test_label
    
    def __len__(self):
        if self.mode == 'train':
            data = self.train_data
        elif self.mode == 'val':
            data = self.val_data
        elif self.mode == 'test':
            data = self.test_data
        else:
            data = self.test_data
        if len(data) < self.seq_len + self.pred_len:
            return 0
        return (len(data) - self.seq_len - self.pred_len) // self.step + 1
    
    def __getitem__(self, index):
        idx = index * self.step
        if self.mode == 'train':
            data = self.train_data
            X = np.float32(data[idx: idx + self.seq_len])
            y = np.float32(data[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            return X, y
        elif self.mode == 'val':
            data = self.test_data
            X = np.float32(data[idx: idx + self.seq_len])
            y = np.float32(data[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            return X, y
        elif self.mode == 'test':
            data = self.test_data
            X = np.float32(data[idx: idx + self.seq_len])
            y = np.float32(data[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            label = np.float32(self.test_label[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            return X, y, label
        else:
            data = self.test_data
            X = np.float32(data[idx: idx + self.seq_len])
            y = np.float32(data[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            label = np.float32(self.test_label[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            return X, y, label
    
class SWAT_data_loader(Dataset):
    def __init__(self, root_path, seq_len, pred_len, step=1, mode='train'):
        self.mode = mode
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.step = step
        self.scaler = StandardScaler()

        train_df = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        test_df = pd.read_csv(os.path.join(root_path, 'swat2.csv'))

        train_data = train_df.values[:, :-1]
        test_data = test_df.values[:, :-1]
        test_label = test_df.values[:, -1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)

        self.train_data = train_data
        self.test_data = test_data
        self.val_data = test_data
        self.test_label = test_label
    
    def __len__(self):
        if self.mode == 'train':
            data = self.train_data
        elif self.mode == 'val':
            data = self.val_data
        elif self.mode == 'test':
            data = self.test_data
        else:
            data = self.test_data
        if len(data) < self.seq_len + self.pred_len:
            return 0
        return (len(data) - self.seq_len - self.pred_len) // self.step + 1

    def __getitem__(self, index):
        idx = index * self.step
        if self.mode == 'train':
            data = self.train_data
            X = np.float32(data[idx: idx + self.seq_len])
            y = np.float32(data[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            return X, y
        elif self.mode == 'val':
            data = self.test_data
            X = np.float32(data[idx: idx + self.seq_len])
            y = np.float32(data[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            return X, y
        elif self.mode == 'test':
            data = self.test_data
            X = np.float32(data[idx: idx + self.seq_len])
            y = np.float32(data[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            label = np.float32(self.test_label[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            return X, y, label
        else:
            data = self.test_data
            X = np.float32(data[idx: idx + self.seq_len])
            y = np.float32(data[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            label = np.float32(self.test_label[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            return X, y, label

class CICIDS_data_loader(Dataset):
    def __init__(self, root_path, seq_len, pred_len, step=1, mode='train'):
        self.mode = mode
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.step = step
        self.scaler = StandardScaler()

        train_data = np.load(os.path.join(root_path, 'CICIDS_train.npy'))
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)

        test_data = np.load(os.path.join(root_path, 'CICIDS_test.npy'))
        test_data = self.scaler.transform(test_data)
        test_label = np.load(os.path.join(root_path, 'CICIDS_test_label.npy'))

        self.train_data = train_data
        self.test_data = test_data
        self.val_data = test_data
        self.test_label = test_label

    def __len__(self):
        if self.mode == 'train':
            data = self.train_data
        elif self.mode == 'val':
            data = self.val_data
        elif self.mode == 'test':
            data = self.test_data
        else:
            data = self.test_data
        if len(data) < self.seq_len + self.pred_len:
            return 0
        return (len(data) - self.seq_len - self.pred_len) // self.step + 1

    def __getitem__(self, index):
        idx = index * self.step
        if self.mode == 'train':
            data = self.train_data
            X = np.float32(data[idx: idx + self.seq_len])
            y = np.float32(data[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            return X, y
        elif self.mode == 'val':
            data = self.val_data
            X = np.float32(data[idx: idx + self.seq_len])
            y = np.float32(data[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            return X, y
        elif self.mode == 'test':
            data = self.test_data
            X = np.float32(data[idx: idx + self.seq_len])
            y = np.float32(data[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            label = np.float32(self.test_label[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            return X, y, label
        else:
            data = self.data
            X = np.float32(data[idx: idx + self.seq_len])
            y = np.float32(data[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            label = np.float32(self.label[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            return X, y, label

class KDD99_data_loader(Dataset):
    def __init__(self, root_path, seq_len, pred_len, step=1, mode='train'):
        self.mode = mode
        self.seq_len = seq_len 
        self.pred_len = pred_len
        self.step = step
        self.scaler = StandardScaler()

        train_data = np.load(os.path.join(root_path, 'KDD99_train.npy'))
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)

        test_data = np.load(os.path.join(root_path, 'KDD99_test.npy'))
        test_data = self.scaler.transform(test_data)
        test_label = np.load(os.path.join(root_path, 'KDD99_test_label.npy'))

        self.train_data = train_data
        self.test_data = test_data
        self.val_data = test_data
        self.test_label = test_label

    def __len__(self):
        if self.mode == 'train':
            data = self.train_data
        elif self.mode == 'val':
            data = self.val_data
        elif self.mode == 'test':
            data = self.test_data
        else:
            data = self.test_data
        if len(data) < self.seq_len + self.pred_len:
            return 0
        return (len(data) - self.seq_len - self.pred_len) // self.step + 1
    
    def __getitem__(self, index):
        idx = index * self.step
        if self.mode == 'train':
            data = self.train_data
            X = np.float32(data[idx: idx + self.seq_len])
            y = np.float32(data[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            return X, y
        elif self.mode == 'val':
            data = self.test_data
            X = np.float32(data[idx: idx + self.seq_len])
            y = np.float32(data[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            return X, y
        elif self.mode == 'test':
            data = self.test_data
            X = np.float32(data[idx: idx + self.seq_len])
            y = np.float32(data[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            label = np.float32(self.test_label[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            return X, y, label
        else:
            data = self.test_data
            X = np.float32(data[idx: idx + self.seq_len])
            y = np.float32(data[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            label = np.float32(self.test_label[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            return X, y, label

class NIPS_TS_Creditcard_data_loader(Dataset):
    def __init__(self, root_path, seq_len, pred_len, step=1, mode='train'):
        self.mode = mode
        self.seq_len = seq_len 
        self.pred_len = pred_len
        self.step = step
        self.scaler = StandardScaler()

        data = np.load(os.path.join(root_path, 'NIPS_TS_creditcard_train.npy'))
        n = len(data)
        train_data = data[:int(n * 0.5)]
        self.train_data = self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)

        test_data = data[int(n * 0.5):]
        test_data = self.scaler.transform(test_data)
        test_label = np.load(os.path.join(root_path, 'NIPS_TS_creditcard_test_label.npy'))[int(n * 0.5):]

        self.train_data = train_data
        self.test_data = test_data
        self.val_data = test_data
        self.test_label = test_label

    def __len__(self):
        if self.mode == 'train':
            data = self.train_data
        elif self.mode == 'val':
            data = self.val_data
        elif self.mode == 'test':
            data = self.test_data
        else:
            data = self.test_data
        if len(data) < self.seq_len + self.pred_len:
            return 0
        return (len(data) - self.seq_len - self.pred_len) // self.step + 1

    def __getitem__(self, index):
        idx = index * self.step
        if self.mode == 'train':
            data = self.train_data
            X = np.float32(data[idx: idx + self.seq_len])
            y = np.float32(data[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            return X, y
        elif self.mode == 'val':
            data = self.test_data
            X = np.float32(data[idx: idx + self.seq_len])
            y = np.float32(data[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            return X, y
        elif self.mode == 'test':
            data = self.test_data
            X = np.float32(data[idx: idx + self.seq_len])
            y = np.float32(data[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            label = np.float32(self.test_label[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            return X, y, label
        else:
            data = self.test_data
            X = np.float32(data[idx: idx + self.seq_len])
            y = np.float32(data[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            label = np.float32(self.test_label[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            return X, y, label

class NIPS_TS_GECCO_data_loader(Dataset):
    def __init__(self, root_path, seq_len, pred_len, step=1, mode='train'):
        self.mode = mode
        self.seq_len = seq_len 
        self.pred_len = pred_len
        self.step = step
        self.scaler = StandardScaler()

        train_data = np.load(os.path.join(root_path, 'NIPS_TS_Water_train.npy'))
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)

        test_data = np.load(os.path.join(root_path, 'NIPS_TS_Water_test.npy'))
        test_data = self.scaler.transform(test_data)
        test_label = np.load(os.path.join(root_path, 'NIPS_TS_Water_test_label.npy'))

        self.train_data = train_data
        self.test_data = test_data
        self.val_data = test_data
        self.test_label = test_label

    def __len__(self):
        if self.mode == 'train':
            data = self.train_data
        elif self.mode == 'val':
            data = self.val_data
        elif self.mode == 'test':
            data = self.test_data
        else:
            data = self.test_data
        if len(data) < self.seq_len + self.pred_len:
            return 0
        return (len(data) - self.seq_len - self.pred_len) // self.step + 1

    def __getitem__(self, index):
        idx = index * self.step
        if self.mode == 'train':
            data = self.train_data
            X = np.float32(data[idx: idx + self.seq_len])
            y = np.float32(data[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            return X, y
        elif self.mode == 'val':
            data = self.val_data
            X = np.float32(data[idx: idx + self.seq_len])
            y = np.float32(data[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            return X, y
        elif self.mode == 'test':
            data = self.test_data
            X = np.float32(data[idx: idx + self.seq_len])
            y = np.float32(data[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            label = np.float32(self.test_label[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            return X, y, label
        else:
            data = self.test_data
            X = np.float32(data[idx: idx + self.seq_len])
            y = np.float32(data[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            label = np.float32(self.test_label[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            return X, y, label

class NIPS_TS_SWAN_data_loader(Dataset):
    def __init__(self, root_path, seq_len, pred_len, step=1, mode='train'):
        self.mode = mode
        self.seq_len = seq_len
        self.pred_len = pred_len 
        self.step = step
        self.scaler = StandardScaler()

        train_data = np.load(os.path.join(root_path, 'NIPS_TS_Swan_train.npy'))
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)

        test_data = np.load(os.path.join(root_path, 'NIPS_TS_Swan_test.npy'))
        test_data = self.scaler.transform(test_data)
        test_label = np.load(os.path.join(root_path, 'NIPS_TS_Swan_test_label.npy'))

        self.train_data = train_data
        self.test_data = test_data
        self.val_data = test_data
        self.test_label = test_label

    def __len__(self):
        if self.mode == 'train':
            data = self.train_data
        elif self.mode == 'val':
            data = self.val_data
        elif self.mode == 'test':
            data = self.test_data
        else:
            data = self.test_data
        if len(data) < self.seq_len + self.pred_len:
            return 0
        return (len(data) - self.seq_len - self.pred_len) // self.step + 1
    
    def __getitem__(self, index):
        idx = index * self.step
        if self.mode == 'train':
            data = self.train_data
            X = np.float32(data[idx: idx + self.seq_len])
            y = np.float32(data[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            return X, y
        elif self.mode == 'val':
            data = self.test_data
            X = np.float32(data[idx: idx + self.seq_len])
            y = np.float32(data[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            return X, y
        elif self.mode == 'test':
            data = self.test_data
            X = np.float32(data[idx: idx + self.seq_len])
            y = np.float32(data[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            label = np.float32(self.test_label[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            return X, y, label
        else:
            data = self.test_data
            X = np.float32(data[idx: idx + self.seq_len])
            y = np.float32(data[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            label = np.float32(self.test_label[idx + self.seq_len: idx + self.seq_len + self.pred_len])
            return X, y, label