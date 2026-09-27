import sys
sys.path.append('..')
sys.path.append('.')

import pandas as pd
import numpy as np
import os

data_path = './datasets/CICIDS.csv'
save_dir = './datasets/CICIDS'
save_title = save_dir.split('/')[-1]

df_raw = pd.read_csv(data_path)

df_wide = df_raw.pivot(index='date', columns='cols', values='data')

cols = list(df_wide.columns)
if 'label' in cols:
     cols.remove('label')
     cols.append('label')
     df_wide = df_wide[cols]

n = len(df_wide)
train_size = int(n * 0.5)

train_data = df_wide.iloc[:train_size, :-1].values
test_data = df_wide.iloc[train_size:, :-1].values
test_label = df_wide.iloc[train_size:, -1].values

os.makedirs(save_dir, exist_ok=True)

np.save(os.path.join(save_dir, f'{save_title}_train.npy'), train_data)
np.save(os.path.join(save_dir, f'{save_title}_test.npy'), test_data)
np.save(os.path.join(save_dir, f'{save_title}_test_label.npy'), test_label)
# np.save('./datasets/KDD99/KDD99_train_data.npy', train_data)
# np.save('./datasets/KDD99/KDD99_test_data.npy', test_data)
# np.save('./datasets/KDD99/KDD99_test_label.npy', test_label)


print(f'df_wide.shape: {df_wide.shape}')
print(f'df_wide.columns: {df_wide.columns}')
print(f'df_wide.head: {df_wide.head()}')

info_path = os.path.join(save_dir, 'info.txt')
with open(info_path, 'w') as f:
     f.write("Dataset splitting info (50% train, 50% test)\n")
     f.write("=" * 50 + "\n\n")
     
     f.write(f"Train data shape: {train_data.shape}\n")
     f.write(f"Test data shape:  {test_data.shape}\n")
     f.write(f"Test label shape: {test_label.shape}\n")
     f.write(f'Features: {train_data.shape[1]}\n\n')
     
     f.write("First 5 rows of train data (features only):\n")
     
     np.set_printoptions(threshold=np.inf, linewidth=200)
     head_str = np.array2string(train_data[:5], separator=', ')
     f.write(head_str + "\n")