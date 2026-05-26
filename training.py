import numpy as np
import torch
import platform
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


LR_BASE   = 1e-3     # learning rate for LSTM and ANN
LR_META   = 1e-3     # learning rate for meta-model
EPOCHS_BASE = 50     # epochs for each base model
EPOCHS_META = 30     # epochs for meta-model

if platform.system() == "Darwin":
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def one_epoch(model, loader, optimizer, criterion, device):
    """Run one full pass over training data. Returns average loss."""
    model.train()
    total_loss = 0.0
    
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(X_batch)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    """Run model in eval mode (dropout off). Returns avg loss."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            total_loss += loss.item() * len(X_batch)
            
    return total_loss / len(loader.dataset)