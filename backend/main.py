import time
from discovery import start_discovery, get_devices

def main():
    # Start device discovery
    start_discovery()
    
    # Poll for devices
    while True:
        devices = get_devices()
        if len(devices) >= 2:
            print(f"Total devices discovered: {len(devices)}")
            break
        print(f"Devices discovered so far: {devices}")
        time.sleep(5)  # Polling interval

if __name__ == '__main__':
    main()
