import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import pandas
import numpy as np

class MLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)

if __name__ == '__main__':

    torch.manual_seed(42)
    np.random.seed(42)

    dataframe = pandas.read_csv("data/breast-cancer/data.csv", index_col="id")
    y = dataframe.diagnosis.values
    x = dataframe.drop("diagnosis", axis=1)

    xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

    scaler = pp.MinMaxScaler()
    scaler.fit(x)
    xtrain = scaler.transform(xtrain)
    xtest = scaler.transform(xtest)

    xtrain_t = torch.tensor(xtrain, dtype=torch.float32)
    xtest_t = torch.tensor(xtest,  dtype=torch.float32)
    ytrain_t = torch.tensor(ytrain, dtype=torch.long)
    ytest_t = torch.tensor(ytest,  dtype=torch.long)

    train_loader = DataLoader(TensorDataset(xtrain_t, ytrain_t), batch_size=5, shuffle=True)




    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(x.shape[1]).to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters())

    epochs = 10
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(xtest_t.to(device)).argmax(dim=1)
            val_acc = (val_preds == ytest_t.to(device)).to(torch.float32).mean().item()
            train_preds = model(xtrain_t.to(device)).argmax(dim=1)
            train_loss = criterion(model(xtrain_t.to(device)), ytrain_t.to(device)).item()

        print(f"Epoch {epoch+1}/{epochs}  loss={train_loss:.4f}  val_acc={val_acc*100:.1f}%")

    print(f"\nTotal accuracy: {val_acc*100:.1f}%")

    torch.save(model.state_dict(), "data/breast-cancer/cancer_mlp.pth")
