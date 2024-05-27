# main.py
import time
import threading
from discovery import start_discovery, get_devices
import torch.nn as nn
from training import run_training
from torchvision import datasets, transforms

def input_thread(stop_event):
    while not stop_event.is_set():
        user_input = input("Type 'start' to begin training or 'status' to see connected devices: ").strip().lower()
        if user_input == 'start':
            stop_event.set()
        elif user_input == 'status':
            devices = get_devices()
            print(f"Devices discovered so far: {devices}")
        else:
            print("Invalid command. Type 'start' to begin training or 'status' to see connected devices.")

def main():
    # Start device discovery
    start_discovery()

    # Create an event to stop the input thread
    stop_event = threading.Event()

    # Start the input thread
    input_thread_instance = threading.Thread(target=input_thread, args=(stop_event,))
    input_thread_instance.start()

    # Wait for the user to trigger the start event
    while not stop_event.is_set():
        time.sleep(1)

    # Get the list of discovered devices
    devices = get_devices()
    world_size = len(devices)

    # If not enough devices are found, exit
    if world_size < 2:
        print("Not enough devices discovered. Exiting.")
        return

    # Ask the user for the model type
    print("Select a model to train:")
    print("1. Simple Linear Model")
    print("2. Simple Convolutional Model")
    print("3. Custom Model")
    model_choice = input("Enter the number of the model you want to train: ").strip()

    if model_choice == '1':
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    elif model_choice == '2':
        model = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    elif model_choice == '3':
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    else:
        print("Invalid choice. Exiting.")
        return

    # Define the dataset
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.FakeData(transform=transform)  # Using FakeData for simplicity

    # Run distributed training
    run_training(world_size, model, dataset)

if __name__ == '__main__':
    main()
