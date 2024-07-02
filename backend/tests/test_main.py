from fastapi.testclient import TestClient
from app.main import app
import time

client = TestClient(app)

def test_register_device():
    response = client.post("/devices/register", json={"id": "1", "name": "Device1", "memory": 16, "cpu": 8, "gpu": 1})
    assert response.status_code == 200
    assert response.json() == {"message": "Device registered successfully"}

def test_list_devices():
    response = client.get("/devices")
    assert response.status_code == 200
    assert response.json() == [{"id": "1", "name": "Device1", "memory": 16, "cpu": 8, "gpu": 1}]

def test_submit_task():
    response = client.post("/tasks/submit", json={"id": "1", "model_architecture": "Model1", "dataset": "Dataset1", "devices": ["1"]})
    assert response.status_code == 200
    task_id = response.json().get("task_id")
    assert task_id is not None

    # Wait a bit for the task to be processed
    time.sleep(1)

    # Check the status of the task
    status_response = client.get(f"/tasks/{task_id}/status")
    assert status_response.status_code == 200
    assert "status" in status_response.json()
    if status_response.json()["status"] == "completed":
        assert "result" in status_response.json()

def test_list_tasks():
    response = client.get("/tasks")
    assert response.status_code == 200
    assert len(response.json()) > 0
