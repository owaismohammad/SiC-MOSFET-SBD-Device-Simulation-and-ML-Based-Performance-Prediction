import torch
import torch.nn as nn
from sklearn.metrics import r2_score

def train_step(train_dataloader: torch.utils.data.DataLoader,
               model: nn.Module,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    train_loss = 0
    all_preds = []
    all_labels = []

    model.train()
    for X, y in train_dataloader:
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        all_preds.append(y_pred.detach().cpu())
        all_labels.append(y.detach().cpu())

    train_loss /= len(train_dataloader)

    # Combine all predictions and labels
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # Compute R² score as "accuracy"
    train_acc = r2_score(all_labels, all_preds)

    return train_loss, train_acc


def test_step(model: nn.Module,
              loss_fn: nn.Module,
              test_dataloader: torch.utils.data.DataLoader,
              device: torch.device):
    model.eval()
    test_loss = 0
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)

            test_loss += loss.item()
            all_preds.append(y_pred.cpu())
            all_labels.append(y.cpu())

    test_loss /= len(test_dataloader)

    # Combine all predictions and labels
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # Compute R² score
    test_acc = r2_score(all_labels, all_preds)

    return test_loss, test_acc


def train(model: nn.Module,
          loss_fn: nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader):

    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    for epoch in range(epochs):
        train_loss, train_acc = train_step(train_dataloader, model, loss_fn, optimizer, device)
        test_loss, test_acc = test_step(model, loss_fn, test_dataloader, device)

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        print(f"EPOCH: {epoch+1}\n"
              f" Train Loss : {train_loss:.4f}  | Train R² : {train_acc:.4f}\n"
              f" Test Loss  : {test_loss:.4f}   | Test R²  : {test_acc:.4f}")
    return results
