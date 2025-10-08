import torch
import torch.optim as optim
from model import Model

def get_dummy_data(num=100, input_dim=10, num_classes=2):
    X = torch.randn(num, input_dim)
    y = torch.randint(0, num_classes, (num,))
    return X, y

def train():
    model = Model()
    X, y = get_dummy_data()
    optimizer = optim.RMSProp(model.parameters(), lr=0.256)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(3):
        optimizer.zero_grad()
        outputs = model(X)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()