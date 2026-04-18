"""
EALSTM_helper.py
Helper functions and classes for the Entity-Aware LSTM (EA-LSTM) model.
Based on Kratzert et al. (2019) - Towards learning universal, regional, and 
local hydrological behaviors via machine learning applied to large-sample datasets.

Implementation note: the sequence processing uses PyTorch's built-in optimized 
LSTM kernel rather than a manual Python timestep loop for performance. The 
entity-aware concept is preserved — static catchment attributes modulate the 
final hidden state via a learned input gate, conditioning predictions on basin 
characteristics.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# --- EA-LSTM Architecture ---

class EALSTM(nn.Module):
    """
    Entity-Aware LSTM (Kratzert et al., 2019)

    Uses PyTorch's built-in LSTM for fast sequence processing, with a
    separate entity-aware gate computed from static catchment attributes
    that modulates the final hidden state. This allows the model to condition
    its predictions on basin characteristics such as area and elevation.

    Parameters:
        dynamic_input_size : number of dynamic (time-varying) features
        static_input_size  : number of static catchment attributes
        hidden_size        : number of LSTM hidden units
        dropout            : dropout rate
    """
    def __init__(self, dynamic_input_size, static_input_size, hidden_size, dropout=0.0):
        super(EALSTM, self).__init__()

        self.hidden_size = hidden_size

        # Built-in optimized LSTM for dynamic inputs
        self.lstm = nn.LSTM(
            input_size  = dynamic_input_size,
            hidden_size = hidden_size,
            num_layers  = 1,
            batch_first = True
        )

        # Entity-aware gate driven by static attributes only
        # Computed once per basin, conditions the hidden state
        self.W_i = nn.Linear(static_input_size, hidden_size)

        # Output layer
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)

    def forward(self, x_dynamic, x_static):
        """
        x_dynamic : (batch_size, seq_len, dynamic_input_size)
        x_static  : (batch_size, static_input_size)
        returns   : (batch_size, 1)
        """
        # Run optimized LSTM over full sequence
        lstm_out, _ = self.lstm(x_dynamic)  # (batch, seq_len, hidden)

        # Take last timestep hidden state
        h = lstm_out[:, -1, :]              # (batch, hidden)

        # Compute entity-aware gate from static attributes
        # This is the basin-specific modulation — fixed per basin
        i = torch.sigmoid(self.W_i(x_static))  # (batch, hidden)

        # Modulate hidden state by basin-specific gate
        h = h * i

        out = self.fc(self.dropout(h))
        return out


# --- Dataset ---

class EASequenceDataset(Dataset):
    """
    PyTorch Dataset for EA-LSTM.
    Returns dynamic sequences, static attributes, and target values.
    """
    def __init__(self, X_dynamic, X_static, y):
        self.X_dynamic = torch.tensor(X_dynamic, dtype=torch.float32)
        self.X_static  = torch.tensor(X_static,  dtype=torch.float32)
        self.y         = torch.tensor(y,          dtype=torch.float32).unsqueeze(1)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X_dynamic[idx], self.X_static[idx], self.y[idx]


# --- Sequence Creation ---

def make_sequences_with_static(df, date_col, dynamic_cols, static_cols,
                                target_col, lookback, gage_col='gage_id'):
    """
    Create lookback sequences for EA-LSTM, processing each site separately
    to avoid sequences that bleed across basin boundaries.
    
    Parameters:
        df           : merged DataFrame with all sites
        date_col     : name of date column
        dynamic_cols : list of time-varying feature column names
        static_cols  : list of static attribute column names
        target_col   : name of target column
        lookback     : number of past timesteps per sequence
        gage_col     : name of gage ID column
    
    Returns:
        X_dynamic : (n_samples, lookback, n_dynamic_features)
        X_static  : (n_samples, n_static_features)
        y         : (n_samples,)
        dates     : (n_samples,) prediction dates
    """
    all_X_dynamic, all_X_static, all_y, all_dates = [], [], [], []
    
    for gage_id in df[gage_col].unique():
        site_df = df[df[gage_col] == gage_id].copy().reset_index(drop=True)
        
        # Static attributes are constant for this site — take first row
        static_vals = site_df[static_cols].iloc[0].values.astype(np.float32)
        
        dynamic = site_df[dynamic_cols].values.astype(np.float32)
        target  = site_df[target_col].values.astype(np.float32)
        dates   = site_df[date_col].values
        
        for i in range(lookback, len(site_df)):
            all_X_dynamic.append(dynamic[i - lookback:i])
            all_X_static.append(static_vals)
            all_y.append(target[i])
            all_dates.append(dates[i])
    
    return (
        np.array(all_X_dynamic),
        np.array(all_X_static),
        np.array(all_y),
        np.array(all_dates)
    )


# --- Scaling Helpers ---

def scale_dynamic(df, dynamic_cols, feature_scaler):
    """Apply pre-fitted feature scaler to dynamic columns, return scaled df."""
    df = df.copy()
    df[dynamic_cols] = feature_scaler.transform(df[dynamic_cols])
    return df


def scale_static(df, static_cols, static_scaler):
    """Apply pre-fitted scaler to static columns, return scaled df."""
    df = df.copy()
    df[static_cols] = static_scaler.transform(df[static_cols])
    return df


def scale_target(df, target_col, target_scaler):
    """Apply pre-fitted target scaler, return scaled df."""
    df = df.copy()
    df[target_col] = target_scaler.transform(df[[target_col]])
    return df


# --- Evaluation ---

def evaluate(model, criterion, device, loader):
    """
    Evaluate EA-LSTM on a dataloader.
    
    Returns:
        mean_loss  : average loss over all batches
        all_preds  : numpy array of scaled predictions
        all_obs    : numpy array of scaled observations
    """
    model.eval()
    all_losses, all_preds, all_obs = [], [], []
    
    with torch.no_grad():
        for xb_dynamic, xb_static, yb in loader:
            xb_dynamic = xb_dynamic.to(device)
            xb_static  = xb_static.to(device)
            yb         = yb.to(device)
            
            pred = model(xb_dynamic, xb_static)
            loss = criterion(pred, yb)
            
            all_losses.append(loss.item())
            all_preds.append(pred.cpu().numpy())
            all_obs.append(yb.cpu().numpy())
    
    mean_loss = float(np.mean(all_losses))
    all_preds = np.concatenate(all_preds).ravel()
    all_obs   = np.concatenate(all_obs).ravel()
    
    return mean_loss, all_preds, all_obs
