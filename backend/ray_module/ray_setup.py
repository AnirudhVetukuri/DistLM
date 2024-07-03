import ray
import socket

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.254.254.254', 1))
        local_ip = s.getsockname()[0]
    except Exception:
        local_ip = '127.0.0.1'
    finally:
        s.close()
    return local_ip

def check_cluster(ip, port):
    try:
        # Try to create a socket connection to the head node
        sock = socket.create_connection((ip, port), timeout=2)
        sock.close()
        return True
    except (socket.timeout, ConnectionRefusedError):
        return False

def initialize_ray():
    local_ip = get_local_ip()
    head_port = 6379  # Use the standard Ray port

    if check_cluster(local_ip, head_port):
        print("Connecting to existing Ray cluster...")
        ray.init(address=f"{local_ip}:{head_port}", ignore_reinit_error=True)
    else:
        print("Starting a new Ray head node...")
        ray.init(ignore_reinit_error=True)

if __name__ == "__main__":
    initialize_ray()
