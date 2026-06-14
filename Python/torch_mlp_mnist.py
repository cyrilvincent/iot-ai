import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

with np.load("data/mnist/mnist.npz", allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test,  y_test  = f['x_test'],  f['y_test']

x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0

x_train = x_train.reshape(-1, 28*28)
x_test  = x_test.reshape(-1, 28*28)

x_train_t = torch.tensor(x_train)
x_test_t  = torch.tensor(x_test)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

train_loader = DataLoader(TensorDataset(x_train_t, y_train_t), batch_size=10, shuffle=True)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = MLP().to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

epochs = 5
for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_out   = model(x_test_t.to(device))
        val_preds = val_out.argmax(dim=1)
        val_acc   = (val_preds == y_test_t.to(device)).sum().item() / len(y_test_t)
        val_loss  = criterion(val_out, y_test_t.to(device)).item()

    print(f"Epoch {epoch+1}/{epochs}  loss={val_loss:.4f}  val_acc={val_acc*100:.2f}%")

torch.save(model.state_dict(), "data/mnist/mnist_mlp.pth")

model.eval()
with torch.no_grad():
    predicted = model(x_test_t.to(device)).cpu().numpy()

print(y_test[:10], predicted[:10], np.argmax(predicted[:10], axis=1))

predicted_classes = np.argmax(predicted, axis=1)
misclass = (y_test != predicted_classes)

x_test_img      = x_test.reshape(-1, 28, 28)
misclass_images = x_test_img[misclass]
misclass_pred   = predicted_classes[misclass]
misclass_true   = y_test[misclass]

select = np.random.randint(misclass_images.shape[0], size=12)

for index, value in enumerate(select):
    plt.subplot(3, 4, index + 1)
    plt.axis('off')
    plt.imshow(misclass_images[value], cmap=plt.cm.gray_r, interpolation="nearest")
    plt.title(f'Pred:{misclass_pred[value]} True:{misclass_true[value]}')

plt.tight_layout()
plt.show()
