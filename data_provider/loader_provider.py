# -*- coding = utf-8 -*-
# @Time: 2025/10/18 11:05
# @Author: wisehone
# @File: loader_provider.py
# @SoftWare: PyCharm
import sys
sys.path.append('..')
sys.path.append('.')


import torch
from data_provider.data_factory import *
from torch.utils.data import DataLoader
from config import dataset2path

def get_data_loader(root_path, batch_size, seq_len, pred_len=1, step=1, mode='train', dataset='KDD', num_workers=8):
    assert dataset in dataset2path, 'dataset doesn\'t found'

    if root_path is None or root_path == '':
        root_path = dataset2path[dataset]

    if dataset == 'MSL':
        dataset_obj = MSL_data_loader(root_path, seq_len, pred_len, step, mode)
    elif dataset == 'PSM':
        dataset_obj = PSM_data_loader(root_path, seq_len, pred_len, step, mode)
    elif dataset == 'SMAP':
        dataset_obj = SMAP_data_loader(root_path, seq_len, pred_len, step, mode)
    elif dataset == 'SMD':
        dataset_obj = SMD_data_loader(root_path, seq_len, pred_len, step, mode)
    elif dataset == 'SWAT':
        dataset_obj = SWAT_data_loader(root_path, seq_len, pred_len, step, mode)
    elif dataset == 'Creditcard':
        dataset_obj = NIPS_TS_Creditcard_data_loader(root_path, seq_len, pred_len, step, mode)
    elif dataset == 'GECCO':
        dataset_obj = NIPS_TS_GECCO_data_loader(root_path, seq_len, pred_len, step, mode)
    elif dataset == 'SWAN':
        dataset_obj = NIPS_TS_SWAN_data_loader(root_path, seq_len, pred_len, step, mode)
    elif dataset == 'KDD99':
        dataset_obj = KDD99_data_loader(root_path, seq_len, pred_len, step, mode)
    elif dataset == 'CICIDS':
        dataset_obj = CICIDS_data_loader(root_path, seq_len, pred_len, step, mode)
    else:
        raise ValueError(f'Unsupported dataset: {dataset}')
    
    shuffle = False

    loader = DataLoader(
        dataset=dataset_obj,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True
    )
    return dataset_obj, loader

if __name__ == '__main__':
    dataset, loader = get_data_loader(None, 4, 100, 20, step=1, mode='test', dataset='CICIDS')
    sm = 0
    for i, (x, y,  label) in enumerate(loader):   # 测试时返回 (X, y, label)
        print(x.shape, label.shape)
        # sm += label.sum()
        # print('sum: ', label.sum())
        break
    print(sm)

    for batch_x, batch_y, *_ in loader:
        print(f'batch_x.shape: {batch_x.shape}, batch_y.shape: {batch_y.shape}')
        break