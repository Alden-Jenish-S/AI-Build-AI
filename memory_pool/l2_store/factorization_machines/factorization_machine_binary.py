import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class FactorizationMachine(nn.Module):
    def __init__(self, n_features: int, k: int = 10):
        super().__init__()
        self.n_features = n_features
        self.k = k
        self.linear = nn.Linear(n_features, 1, bias=True)
        self.V = nn.Embedding(n_features, k)
        nn.init.normal_(self.linear.weight, std=0.01)
        nn.init.normal_(self.V.weight, std=0.01)

    def forward(self, x):
        linear_part = self.linear(x).squeeze(1)
        x_v = torch.matmul(x, self.V.weight)
        inter_part1 = torch.pow(x_v, 2)
        inter_part2 = torch.matmul(torch.pow(x, 2), torch.pow(self.V.weight, 2))
        interaction = 0.5 * torch.sum(inter_part1 - inter_part2, dim=1)
        return linear_part + interaction

def _build_preprocessor(df):
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "bool", "category"]).columns.tolist()
    numeric = Pipeline([("scaler", StandardScaler())])
    categorical = Pipeline([("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))])
    return ColumnTransformer([("num", numeric, num_cols), ("cat", categorical, cat_cols)], remainder="drop")

def train_and_predict(train_df, test_df, target_col="target", k=10, lr=0.01, weight_decay=1e-5, epochs=30, batch_size=1024, device="cpu"):
    y_train = train_df[target_col].values.astype(np.float32)
    X_train = train_df.drop(columns=[target_col])
    X_test = test_df.copy()
    preprocessor = _build_preprocessor(X_train)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    X_train_t = torch.from_numpy(X_train_proc).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    X_test_t = torch.from_numpy(X_test_proc).to(device)
    model = FactorizationMachine(X_train_t.shape[1], k).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_test_t)).cpu().numpy()
    return probs