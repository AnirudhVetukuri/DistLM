import ray
import torch
import torch.optim as optim
from ray_module.data_loader import load_data
from ray_module.model import SimpleCNN

@ray.remote
def train_model(model_architecture, dataset, epochs=1):
    train_loader, _ = load_data()
    model = SimpleCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item()}')
                
    return "Training complete"
