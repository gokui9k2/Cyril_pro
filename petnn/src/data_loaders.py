import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import random 

""""
First version of data loaders for PETNN model. This data Loaser handles time series data, creating sequences for training, validation, and testing. 
but only for the classic PETTN implmentation 
"""

def create_sequences(data, seq_length, pred_length, target_col_name):
    X, y = [], []

    target_index = data.columns.get_loc(target_col_name)
    data_array = data.values.astype(np.float32)

    for i in range(len(data_array) - seq_length - pred_length + 1):
        X.append(data_array[i:i+seq_length])
        y.append(data_array[i+seq_length:i+seq_length+pred_length, target_index])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    if y.ndim == 1:
        y = y.reshape(-1, 1)

    return X, y, target_index


def create_loaders(data, seq_length, pred_length, batch_size=32, target_col_name='target'):

    total_len = len(data)
    train_size = int(0.7 * total_len)
    val_size = int(0.15 * total_len)

    target_col_index = data.columns.get_loc(target_col_name)

    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size:train_size + val_size]
    test_data = data.iloc[train_size + val_size:]

    X_train, y_train, _ = create_sequences(train_data, seq_length, pred_length, target_col_name)
    X_val, y_val, _ = create_sequences(val_data, seq_length, pred_length, target_col_name)
    X_test, y_test, _ = create_sequences(test_data, seq_length, pred_length, target_col_name)

    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=3)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=3)

    return train_loader, val_loader, test_loader, target_col_index


""""
Those data lodaer of for the traning of the masked PETNN model
"""

def create_mae_masked_sequences(data, seq_length=96, patch_size=6, num_masked_patches=4, mask_value=0.0):

    if isinstance(data, pd.DataFrame):
        arr = data.values.astype(np.float32)
    elif isinstance(data, np.ndarray):
        arr = data.astype(np.float32)
    else:
        arr = np.array(data).astype(np.float32)

    X_inputs = []
    X_masks  = []
    Y_targets = []

    num_patches = seq_length // patch_size

    for i in range(len(arr) - seq_length):
        seq = arr[i : i + seq_length]

        patches = [seq[j*patch_size : (j+1)*patch_size] for j in range(num_patches)]

        masked_idx = sorted(random.sample(range(num_patches), num_masked_patches))
        mask_vector = np.zeros(num_patches, dtype=np.int8)
        for idx in masked_idx:
            mask_vector[idx] = 1

        input_patches = []
        for j in range(num_patches):
            if mask_vector[j]:
                input_patches.append(np.full_like(patches[j], mask_value))
            else:
                input_patches.append(patches[j])

        input_seq = np.concatenate(input_patches, axis=0)

        X_inputs.append(input_seq)
        X_masks.append(mask_vector)
        Y_targets.append(seq)

    return np.array(X_inputs), np.array(X_masks), np.array(Y_targets)

def create_loaders_mae(train_scaled, val_scaled, test_scaled, seq_length=96, patch_size=6, num_masked_patches=4, batch_size=64, num_workers=0):

    X_train, mask_train, Y_train = create_mae_masked_sequences(train_scaled, seq_length, patch_size, num_masked_patches,mask_value=0.0)
    X_val, mask_val, Y_val = create_mae_masked_sequences(val_scaled, seq_length, patch_size, num_masked_patches,mask_value=0.0)
    X_test, mask_test, Y_test = create_mae_masked_sequences(test_scaled, seq_length, patch_size, num_masked_patches,mask_value=0.0)

    X_train = torch.FloatTensor(X_train)
    mask_train = torch.LongTensor(mask_train)
    Y_train = torch.FloatTensor(Y_train)

    X_val = torch.FloatTensor(X_val)
    mask_val = torch.LongTensor(mask_val)
    Y_val = torch.FloatTensor(Y_val)

    X_test = torch.FloatTensor(X_test)
    mask_test = torch.LongTensor(mask_test)
    Y_test = torch.FloatTensor(Y_test)

    train_dataset = TensorDataset(X_train, mask_train, Y_train)
    val_dataset   = TensorDataset(X_val, mask_val, Y_val)
    test_dataset  = TensorDataset(X_test, mask_test, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


""" 
Those data ladoer are for the final traning of the model endoer and decoder 
"""


def create_sequences_autoregressive(data, seq_length, pred_length, target_col_name, lookback_window=6):
    X_encoder, X_decoder, y_target = [], [], []

    target_index = data.columns.get_loc(target_col_name)
    data_array = data.values.astype(np.float32)

    for i in range(len(data_array) - seq_length - pred_length):

        encoder_seq = data_array[i : i+seq_length].copy()
        X_encoder.append(encoder_seq)

        y_history = data_array[i+seq_length-lookback_window : i+seq_length, target_index]
        X_decoder.append(y_history)

        y_target.append(data_array[i+seq_length : i+seq_length+pred_length, target_index])

    X_encoder = np.array(X_encoder, dtype=np.float32)
    X_decoder = np.array(X_decoder, dtype=np.float32)
    y_target = np.array(y_target, dtype=np.float32)

    if X_decoder.ndim == 2:
        X_decoder = np.expand_dims(X_decoder, axis=-1)

    if y_target.ndim == 2:
        y_target = np.expand_dims(y_target, axis=-1)

    return X_encoder, X_decoder, y_target

def create_loaders_autoregressive(train_scaled, val_scaled, test_scaled, seq_length, pred_length, target_col_name, batch_size=32, num_workers=0):

    X_enc_train, X_dec_train, y_train = create_sequences_autoregressive(train_scaled, seq_length, pred_length, target_col_name, lookback_window = 6)
    X_enc_val, X_dec_val, y_val = create_sequences_autoregressive(val_scaled, seq_length, pred_length, target_col_name, lookback_window = 6)
    X_enc_test, X_dec_test, y_test = create_sequences_autoregressive(test_scaled, seq_length, pred_length, target_col_name, lookback_window =6 )

    X_enc_train = torch.FloatTensor(X_enc_train)
    X_dec_train = torch.FloatTensor(X_dec_train)
    y_train = torch.FloatTensor(y_train)

    X_enc_val = torch.FloatTensor(X_enc_val)
    X_dec_val = torch.FloatTensor(X_dec_val)
    y_val = torch.FloatTensor(y_val)

    X_enc_test = torch.FloatTensor(X_enc_test)
    X_dec_test = torch.FloatTensor(X_dec_test)
    y_test = torch.FloatTensor(y_test)

    train_dataset = TensorDataset(X_enc_train, X_dec_train, y_train)
    val_dataset = TensorDataset(X_enc_val, X_dec_val, y_val)
    test_dataset = TensorDataset(X_enc_test, X_dec_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader



