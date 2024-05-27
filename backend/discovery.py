import socket
import threading
import time

PORT = 12345
BROADCAST_INTERVAL = 5

devices = set()

def broadcast_presence():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    message = b'Device Presence'
    
    while True:
        sock.sendto(message, ('<broadcast>', PORT))
        time.sleep(BROADCAST_INTERVAL)

def listen_for_devices():
    global devices
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('', PORT))
    
    while True:
        data, addr = sock.recvfrom(1024)
        devices.add(addr[0])
        #print(f'Device discovered: {addr[0]}')
    
def start_discovery():
    thread = threading.Thread(target=broadcast_presence)
    thread.daemon = True
    thread.start()

    listen_thread = threading.Thread(target=listen_for_devices)
    listen_thread.daemon = True
    listen_thread.start()

def get_devices():
    global devices
    return devices
