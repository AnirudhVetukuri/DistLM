import ray
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
from uuid import uuid4
from ray_module.ray_setup import initialize_ray
from ray_module.train import train_model
import os

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(f"{UPLOAD_DIR}/models", exist_ok=True)
os.makedirs(f"{UPLOAD_DIR}/datasets", exist_ok=True)

class Device(BaseModel):
    id: str
    name: str
    memory: int
    cpu: int
    gpu: int

class Task(BaseModel):
    id: str
    model_architecture: str
    dataset: str
    devices: List[str]

devices: List[Device] = []
tasks: List[Task] = []
task_results = {}

initialize_ray()

@app.get("/")
def read_root():
    return {"message": "Welcome to the DistLM backend"}

@app.post("/devices/register")
def register_device(device: Device):
    devices.append(device)
    return {"message": "Device registered successfully"}

@app.get("/devices")
def list_devices():
    return devices

@app.post("/tasks/submit")
def submit_task(task: Task):
    tasks.append(task)
    task_id = str(uuid4())
    result_ref = train_model.remote(task.model_architecture, task.dataset)
    task_results[task_id] = result_ref
    return {"message": "Task submitted successfully", "task_id": task_id}

@app.get("/tasks")
def list_tasks():
    return tasks

@app.get("/tasks/{task_id}/status")
def get_task_status(task_id: str):
    result_ref = task_results.get(task_id)
    if result_ref is None:
        return {"error": "Task not found"}

    try:
        result = ray.get(result_ref, timeout=0)  # Non-blocking check
        return {"task_id": task_id, "status": "completed", "result": result}
    except ray.exceptions.GetTimeoutError:
        return {"task_id": task_id, "status": "in progress"}

@app.post("/upload/model/")
async def upload_model(file: UploadFile = File(...)):
    file_location = f"{UPLOAD_DIR}/models/{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
    return {"info": f"Model '{file.filename}' uploaded successfully"}

@app.post("/upload/dataset/")
async def upload_dataset(file: UploadFile = File(...)):
    file_location = f"{UPLOAD_DIR}/datasets/{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
    return {"info": f"Dataset '{file.filename}' uploaded successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
