import torch 
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import RobustScaler
import random
import matplotlib.pyplot as plt

def one_epoch(model, train_loader, val_loader, optimizer, criterion, device):
    """
    Standard training and validation loop for a single epoch handles forward pass, backprop, gradient clipping, and validation
    """
    model.train()
    train_loss = 0
    total_releases = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, releases = model(data)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()

        if isinstance(releases, torch.Tensor):
             total_releases += releases.item()
        else:
             total_releases += releases

    model.eval()

    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    avg_releases = total_releases / len(train_loader)

    return avg_train_loss, avg_val_loss, avg_releases

def one_epoch_masked_encoder(model_encoder, model_head, train_loader, val_loader, optimizer, device, patch_size, criterion=None):
    """
    Training loop specifically for Masked Autoencoder MAE strategies trains both the Encoder and the Projection Head simultaneously
    """
    model_encoder.train()
    model_head.train()
    train_loss = 0.0
    total_releases = 0.0
    num_train_batches = 0

    for batch in train_loader:
        data_input, mask, target = batch

        data_input = data_input.to(device)
        mask = mask.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        encoder_output, releases = model_encoder(data_input , mask)
        prediction = model_head(encoder_output)

        loss = criterion(prediction, target, mask)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model_encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(model_head.parameters(), max_norm=1.0)

        optimizer.step()
        train_loss += loss.item()
        num_train_batches += 1

        if isinstance(releases, torch.Tensor):
            total_releases += releases.item()
        else:
            total_releases += releases

    model_encoder.eval()
    model_head.eval()
    val_loss = 0.0
    num_val_batches = 0

    with torch.no_grad():
        for batch in val_loader:
      
            data_input, mask, target = batch
        
            data_input = data_input.to(device)
            mask = mask.to(device)
            target = target.to(device)

            encoder_output, _ = model_encoder(data_input , mask)
            prediction = model_head(encoder_output)

            loss = criterion(prediction, target, mask)

            val_loss += loss.item()
            num_val_batches += 1

    avg_train_loss = train_loss / max(1, num_train_batches)
    avg_val_loss = val_loss / max(1, num_val_batches)
    avg_releases = total_releases / max(1, num_train_batches)

    return avg_train_loss, avg_val_loss, avg_releases

def masked_mae_loss(pred, target, mask, patch_size, device):
    """
    Calculates Mean Absolute Error only on the MASKED patches used for self supervised learning where the model reconstructs hidden parts
    """
    mask_ts = mask.repeat_interleave(patch_size, dim=1)
    mask_ts = mask_ts.unsqueeze(-1)

    mae = (pred - target).abs()

    masked_mae = mae * mask_ts

    num_features = pred.shape[-1]
    denom = mask_ts.sum() * num_features + 1e-8
    loss = masked_mae.sum() / denom

    return loss


def evaluate_encoder_decoder(encoder, decoder, data_loader, target_scaler, device, criterion=None):
    """
    Evaluates a full Encoder-Decoder architecture performs Inverse Scaling to report metrics in the original data scale
    """
    encoder.eval()
    decoder.eval()
    
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in data_loader:
            
            if len(batch) == 3:
                data_encoder, _, target = batch
            else:
                data_encoder, target = batch
            
            data_encoder = data_encoder.to(device)
            target = target.to(device)
            
            encoder_output, _ = encoder(data_encoder)
            
            prediction = decoder(encoder_output)
            
            if criterion:
                loss = criterion(prediction, target)
                total_loss += loss.item()
            
            num_batches += 1
            all_predictions.append(prediction.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    pred_shape = predictions.shape
    predictions_flat = predictions.reshape(-1, 1)
    targets_flat = targets.reshape(-1, 1)
    
    predictions_real = target_scaler.inverse_transform(predictions_flat)
    targets_real = target_scaler.inverse_transform(targets_flat)
    
    mse = mean_squared_error(targets_real, predictions_real)
    mae = mean_absolute_error(targets_real, predictions_real)
    rmse = np.sqrt(mse)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    print("-" * 30)
    print(f"Evaluation Results:")
    print(f"Loss (Scaled): {avg_loss:.6f}")
    print(f"MSE (Real)   : {mse:.6f}")
    print(f"RMSE (Real)  : {rmse:.6f}")
    print(f"MAE (Real)   : {mae:.6f}")
    print("-" * 30)
    
    predictions_reshaped = predictions_real.reshape(pred_shape)
    targets_reshaped = targets_real.reshape(pred_shape)
    
    result = {'predictions': predictions_reshaped,
        'targets': targets_reshaped,
        'metrics': {'mse': mse, 'rmse': rmse, 'mae': mae, 'loss': avg_loss}}
    
    return result

def evaluate_model(model, data_loader, target_scaler, device):
    """
    Simplified evaluation for a standard single model
    """
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            outputs, _ = model(X_batch)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(y_batch.numpy())
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    pred_shape = predictions.shape
    predictions_flat = predictions.reshape(-1, 1)
    targets_flat = targets.reshape(-1, 1)

    predictions_original = target_scaler.inverse_transform(predictions_flat)
    targets_original = target_scaler.inverse_transform(targets_flat)

    predictions_original = predictions_original.reshape(pred_shape)
    targets_original = targets_original.reshape(pred_shape)

    mse = mean_squared_error(targets_original, predictions_original)
    mae = mean_absolute_error(targets_original, predictions_original)

    rmse = np.sqrt(mse)
    print(f"MSE={mse}")
    print(f"MAE={mae}")
    print(f"RMSE={rmse}")
    return {'predictions': predictions_original,'targets': targets_original}

def plot_residuals_distribution(result):
    """
    Plots the distribution of residuals between predictions and targets
    """
    n = len(result["predictions"])
    residuals = []
    for i in range(n):
        residual = result["targets"][i] - result["predictions"][i]
        residuals.append(residual)

    residuals_original = np.array(residuals)

    residuals_rounded = np.round(residuals_original).astype(int)

    plt.figure(figsize=(14, 6))
    plt.hist(residuals_rounded.flatten(), bins=50, edgecolor='black', alpha=0.7)

    plt.title("Distribution of Residuals")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return residuals_original

def transfer_weights(path_to_checkpoint, new_model, device):
    """
    Transfers weights from a pre trained checkpoint to a new model
    """
    checkpoint = torch.load(path_to_checkpoint, map_location=device)
    old_state_dict = checkpoint['encoder_state_dict']
    new_state_dict = new_model.state_dict()

    for name, param in old_state_dict.items():

        if "conv.weight" in name:
            clean_weights = param[:, :13, :]

            new_state_dict[name] = clean_weights

        elif name in new_state_dict:

            new_state_dict[name] = param

    new_model.load_state_dict(new_state_dict)

    return new_model

def one_epoch_final(model_encoder, model_head, train_loader, val_loader, optimizer, criterion, device):
    """
    Fine tuning / Linear Probing loop the Encoder is FROZEN only the Head is trained transfer learning setup
    """
    model_encoder.eval()
    model_head.train()

    train_loss = 0
    total_releases = 0

    for batch in train_loader:
        data_encoder = batch[0].to(device)
        target = batch[-1].to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            encoder_output, releases = model_encoder(data_encoder)
        encoder_output = encoder_output.detach().requires_grad_(True)
        prediction = model_head(encoder_output)

        loss = criterion(prediction, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model_head.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()

        if isinstance(releases, torch.Tensor):
            total_releases += releases.item()
        else:
            total_releases += releases
    model_head.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            data_encoder = batch[0].to(device)
            target = batch[-1].to(device)

            encoder_output, _ = model_encoder(data_encoder)
            prediction = model_head(encoder_output)
            loss = criterion(prediction, target)
            val_loss += loss.item()

    return train_loss / len(train_loader), val_loss / len(val_loader), total_releases / len(train_loader)
