from SicsbdMOSFET_model import SicsbdMOSFET
from helper_functions import train
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32


X, y = make_regression(n_samples=1000, n_features=4, n_targets=4, noise=0.1, coef=False)

scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)


X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataoader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)



model = SicsbdMOSFET()
model.to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results = train(
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=100,
    device=device,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataoader
)
