import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np


def loader_tepe(root, batch_size, window_size, stride_size, train_split, label=False):
    data = pd.read_csv('./Dataset/Data/input/TEPE/tepe_data.csv',header=None)
    labels = pd.read_csv("./Dataset/Data/input/TEPE/tepe_label.csv",header=None)
    labels = labels.values
    data = data.astype(float)
    n_sensor = len(data.columns)
    # %%
    feature = data.iloc[:, :53]
    scaler = StandardScaler()
    norm_feature = scaler.fit_transform(feature)
    norm_feature = pd.DataFrame(norm_feature, columns=data.columns)
    norm_feature = norm_feature.dropna(axis=1)

    y = labels.squeeze(-1)
    d = np.diff(y, prepend=0, append=0)
    onsets = np.where(d == 1)[0]
    offsets = np.where(d == -1)[0]
    intervals= list(zip(onsets, offsets))
    N= len(intervals)
    print(N)


    train_df = norm_feature.iloc[:int(train_split * len(data))]
    train_label = labels[:int(train_split * len(data))]
    print('trainset size', train_df.shape, 'anomaly ratio', sum(train_label) / len(train_label))
    # 异常标签占总标签的比例

    val_df = norm_feature.iloc[int(0.6 * len(data)):int(train_split * len(data))]
    val_label = labels[int(0.6 * len(data)):int(train_split * len(data))]

    test_df = norm_feature.iloc[int(train_split * len(data)):]
    test_label = labels[int(train_split * len(data)):]
    print('testset size', test_df.shape, 'anomaly ratio', sum(test_label) / len(test_label))


    if label:
        train_loader = DataLoader(SWat_dataset(train_df, train_label, window_size, stride_size), batch_size=batch_size,
                                  shuffle=False)
    else:
        train_loader = DataLoader(SWat_dataset(train_df, train_label, window_size, stride_size), batch_size=batch_size,
                                  shuffle=True)
    val_loader = DataLoader(SWat_dataset(val_df, val_label, window_size, stride_size), batch_size=batch_size,
                            shuffle=False)
    test_loader = DataLoader(SWat_dataset(test_df, test_label, window_size, stride_size), batch_size=batch_size,
                             shuffle=False)
    return train_loader, val_loader, test_loader, n_sensor





class SWat_dataset(Dataset):
    def __init__(self, df, label, window_size=60, stride_size=10) -> None:
        super(SWat_dataset, self).__init__()
        self.df = df
        self.window_size = window_size
        self.stride_size = stride_size

        self.data, self.idx, self.label = self.preprocess(df, label)
        self.columns = np.append(df.columns, ["Label"])
        self.timeindex = df.index[self.idx]

    def preprocess(self, df, label):
        start_idx = np.arange(0, len(df) - self.window_size, self.stride_size)
        end_idx = np.arange(self.window_size, len(df), self.stride_size)

        label = [0 if sum(label[index:index + self.window_size]) == 0 else 1 for index in
                 start_idx]
        return df.values, start_idx, np.array(label)

    def __len__(self):
        length = len(self.idx)

        return length

    def __getitem__(self, index):
        #  N , K , L , D
        """
        """
        start = self.idx[index]
        end = start + self.window_size
        time=self.timeindex[index]
        data = self.data[start:end].reshape([self.window_size, -1, 1])
        return torch.FloatTensor(data).transpose(0, 1), self.label[index], time

