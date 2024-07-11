from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import ray
import uuid
from datetime import datetime

from ray_module.scheduler import JobScheduler
from ray_module.resource_manager import ResourceManager
from ray_module.token_system import TokenSystem

app = FastAPI(title="DistLM Backend", version="0.1.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
resource_manager = ResourceManager()
job_scheduler = JobScheduler(resource_manager)
token_system = TokenSystem()

# Models
class NodeInfo(BaseModel):
    node_id: str
    hostname: str
    resources: Dict[str, float]  # cpu, memory, gpu
    status: str = "available"
    last_heartbeat: datetime = None
    user_id: str

class JobSubmission(BaseModel):
    model_config: Dict[str, Any]
    data_config: Dict[str, Any]
    requirements: Dict[str, float]
    hyperparameters: Optional[Dict[str, Any]]
    user_id: str

class TokenTransaction(BaseModel):
    user_id: str
    amount: float

# Node Management
@app.post("/nodes/register")
async def register_node(node_info: NodeInfo):
    node_id = str(uuid.uuid4())
    node_info.node_id = node_id
    node_info.last_heartbeat = datetime.now()
    
    # Record initial compute contribution
    token_system.record_compute_contribution(
        node_info.user_id,
        node_id,
        datetime.now(),
        datetime.now(),
        node_info.resources
    )
    
    return {"node_id": node_id, "status": "registered"}

@app.get("/nodes")
async def list_nodes():
    return list(nodes.values())

@app.get("/nodes/{node_id}")
async def get_node(node_id: str):
    if node_id not in nodes:
        raise HTTPException(status_code=404, detail="Node not found")
    return nodes[node_id]

@app.post("/nodes/{node_id}/heartbeat")
async def node_heartbeat(node_id: str, node_info: NodeInfo):
    node_info.last_heartbeat = datetime.now()
    
    # Update compute contribution
    token_system.record_compute_contribution(
        node_info.user_id,
        node_id,
        datetime.now(),
        datetime.now(),
        node_info.resources
    )
    
    return {"status": "ok"}

# Job Management
@app.post("/jobs/submit")
async def submit_job(job: JobSubmission):
    # Check if user has enough tokens
    estimated_cost = token_system.estimate_compute_cost(
        job.requirements,
        24.0  # Assume 24-hour maximum duration for now
    )
    
    if not token_system.check_compute_access(job.user_id, estimated_cost):
        raise HTTPException(
            status_code=402,
            detail="Insufficient tokens for compute access"
        )
    
    # Submit job
    job_id = str(uuid.uuid4())
    job_config = {
        "job_id": job_id,
        **job.dict()
    }
    
    result = await job_scheduler.submit_job(job_config)
    if result:
        # Reserve tokens for the job
        token_system.spend_tokens(job.user_id, estimated_cost)
        return {"job_id": job_id, "status": "submitted"}
    else:
        raise HTTPException(
            status_code=500,
            detail="Failed to submit job"
        )

@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    status = job_scheduler.get_job_status(job_id)
    if status["status"] == "not_found":
        raise HTTPException(status_code=404, detail="Job not found")
    return status

@app.get("/jobs")
async def list_jobs():
    return job_scheduler.list_jobs()

@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    if job_scheduler.cancel_job(job_id):
        return {"status": "cancelled"}
    raise HTTPException(status_code=404, detail="Job not found")

# Token Management
@app.get("/tokens/balance/{user_id}")
async def get_token_balance(user_id: str):
    return {"balance": token_system.get_balance(user_id)}

@app.get("/tokens/history/{user_id}")
async def get_compute_history(user_id: str):
    return {"history": token_system.get_compute_history(user_id)}

@app.post("/tokens/estimate")
async def estimate_compute_cost(
    resources: Dict[str, float],
    duration_hours: float
):
    cost = token_system.estimate_compute_cost(resources, duration_hours)
    return {"estimated_cost": cost}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
