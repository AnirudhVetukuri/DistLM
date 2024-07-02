from pydantic import BaseModel
from typing import List

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
