# training.py
import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main_worker(rank, world_size, model, dataset, epochs=5):
    setup(rank, world_size)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ddp_model = DDP(model)
    
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    print(f"Rank {rank} waiting for start signal")
    while not os.path.exists("start_training.txt"):
        time.sleep(1)
    print(f"Rank {rank} received start signal")

    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(data.size(0), -1).to(device) if data.dim() > 2 else data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print(f'Rank {rank}, Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}')

    cleanup()

def run_training(world_size, model, dataset, epochs=5):
    mp.spawn(main_worker, args=(world_size, model, dataset, epochs), nprocs=world_size, join=True)
