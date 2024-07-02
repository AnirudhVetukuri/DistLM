import ray
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from uuid import uuid4

app = FastAPI()

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

ray.init()

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

@ray.remote
def train_model(model_architecture, dataset):
    # Implement the training logic here
    return f"Training {model_architecture} on {dataset}"

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
