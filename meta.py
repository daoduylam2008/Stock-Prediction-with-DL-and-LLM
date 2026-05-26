import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import training



def train_meta_model(meta_model, lstm_train_preds, ann_train_preds,
                     y_train, lstm_val_preds, ann_val_preds, y_val,
                     epochs=training.EPOCHS_META, lr=training.LR_META):
    """
    Train the stacking meta-model (Linear regression) using the
    base models' predictions as features..
    """
    meta_model = meta_model.to(training.DEVICE)
    optimizer  = torch.optim.Adam(meta_model.parameters(), lr=lr)
    criterion  = nn.MSELoss()

    # Build tiny DataLoaders for the meta-model inputs
    train_ds = TensorDataset(lstm_train_preds.to(training.DEVICE),
                             ann_train_preds.to(training.DEVICE),
                             y_train.to(training.DEVICE))
    val_ds   = TensorDataset(lstm_val_preds.to(training.DEVICE),
                             ann_val_preds.to(training.DEVICE),
                             y_val.to(training.DEVICE))
    train_ld = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_ld   = DataLoader(val_ds,   batch_size=64, shuffle=False)

    best_val   = float("inf")
    best_state = None
    train_losses, val_losses = [], []

    print(f"\n{'='*50}")
    print(f"Training Meta-Model (Stacking Layer)")
    print(f"{'='*50}")

    for epoch in range(1, epochs + 1):
        meta_model.train()
        ep_loss = 0.0
        for lstm_b, ann_b, y_b in train_ld:
            optimizer.zero_grad()
            pred = meta_model(lstm_b, ann_b)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * len(y_b)
        ep_loss /= len(train_ds)

        # Validation
        meta_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for lstm_b, ann_b, y_b in val_ld:
                val_loss += criterion(meta_model(lstm_b, ann_b), y_b).item() * len(y_b)
        val_loss /= len(val_ds)

        train_losses.append(ep_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in meta_model.state_dict().items()}

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Train MSE: {ep_loss:.6f} | Val MSE: {val_loss:.6f}")

    meta_model.load_state_dict(best_state)
    print(f"  Best val MSE: {best_val:.6f}")

    # Print learned weights — shows how much each base model is trusted
    with torch.no_grad():
        w = meta_model.fc.weight.cpu().numpy()[0]
        b = meta_model.fc.bias.cpu().numpy()[0]
        print(f"\n  Meta-model learned: "
              f"β_LSTM={w[0]:.4f}, β_ANN={w[1]:.4f}, β0={b:.4f}")

    return meta_model, train_losses, val_losses