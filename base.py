import torch
import torch.nn as nn
import training

def train(model, train_loader, test_loader,
                     epochs=training.EPOCHS_BASE, lr=training.LR_BASE, name="Model"):
    """
    Train LSTM or ANN using Adam + MSE loss.
    """
    model = model.to(training.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    # ReduceLROnPlateau: halve LR if val loss stalls for 5 epochs
    # This prevents getting stuck in local minima during later training
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state    = None

    print(f"\n{'='*50}")
    print(f"Training {name} on {training.DEVICE}")

    for epoch in range(1, epochs + 1):
        train_loss = training.one_epoch(model, train_loader, optimizer,
                                     criterion, training.DEVICE)
        val_loss   = training.evaluate(model, test_loader, criterion, training.DEVICE)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Save best checkpoint (early stopping equivalent)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

    # Restore best weights
    model.load_state_dict(best_state)
    print(f"  Best val MSE: {best_val_loss:.6f}")
    return model, train_losses, val_losses


def predict(lstm_model, ann_model, loader, device=training.DEVICE):
    """
    Run both base models in inference mode over an entire dataset.
    Returns stacked predictions ready to train the meta-model.
    """
    lstm_model.eval()
    ann_model.eval()

    all_lstm, all_ann, all_y = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            all_lstm.append(lstm_model(X_batch).cpu())
            all_ann.append(ann_model(X_batch).cpu())
            all_y.append(y_batch)

    lstm_preds = torch.cat(all_lstm)   # (N, 1)
    ann_preds  = torch.cat(all_ann)    # (N, 1)
    y_true     = torch.cat(all_y)      # (N, 1)

    return lstm_preds, ann_preds, y_true

