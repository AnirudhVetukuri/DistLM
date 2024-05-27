from fastapi import FastAPI
import ray
import time
import socket
import subprocess
import os

app = FastAPI()
devices = {}

def get_device_id():
    return socket.gethostname()

def register_main_device():
    device_id = get_device_id()
    devices[device_id] = {
        "cpu_cores": 4,  # Example values, adjust as necessary
        "gpu": False,
        "memory_gb": 16,
        "status": "active",
        "last_heartbeat": time.time()
    }
    print(f"Device {device_id} registered with the following details:")
    print(devices[device_id])

@app.on_event("startup")
async def startup_event():
    ray_executable_path = r'C:\Python311\Scripts\ray.exe'
    try:
        ray.init(address='auto', ignore_reinit_error=True)
    except ConnectionError:
        # Start Ray head node
        subprocess.Popen([ray_executable_path, "start", "--head", "--port=6379"])
        time.sleep(5)  # Wait for Ray to start
        try:
            ray.init(address='auto', ignore_reinit_error=True)
        except ConnectionError as e:
            print(f"Failed to initialize Ray after starting head node: {e}")
            raise

    register_main_device()

@app.get("/devices/")
async def get_devices():
    return devices

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
