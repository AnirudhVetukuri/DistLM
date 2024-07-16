from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import ray
import uuid
from datetime import datetime, timedelta

from ray_module.scheduler import JobScheduler
from ray_module.resource_manager import ResourceManager
from ray_module.token_system import TokenSystem
from auth.auth_handler import (
    User, Token, create_access_token, 
    get_current_active_user, authenticate_user,
    create_user, check_permissions
)
from monitoring.metrics import metrics_collector
from config import JWT_CONFIG, API_CONFIG

app = FastAPI(
    title="DistLM Backend",
    version="0.1.0",
    description="Distributed LLM Training Platform"
)

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

# Authentication endpoints
@app.post("/auth/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=JWT_CONFIG["access_token_expire_minutes"])
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role},
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/auth/register", response_model=User)
async def register_user(
    username: str,
    password: str,
    role: str = "user",
    current_user: User = Depends(get_current_active_user)
):
    if not check_permissions(current_user, "admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return create_user(username, password, role)

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
async def register_node(
    node_info: NodeInfo,
    current_user: User = Depends(get_current_active_user)
):
    if not check_permissions(current_user, "node_owner"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Must be a node owner to register nodes"
        )
        
    node_id = str(uuid.uuid4())
    node_info.node_id = node_id
    node_info.last_heartbeat = datetime.now()
    node_info.user_id = current_user.username
    
    # Record initial compute contribution
    token_system.record_compute_contribution(
        node_info.user_id,
        node_id,
        datetime.now(),
        datetime.now(),
        node_info.resources
    )
    
    # Record node metrics
    metrics_collector.record_node_metrics(
        node_id,
        {
            "resources": node_info.resources,
            "status": node_info.status
        }
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
async def node_heartbeat(
    node_id: str,
    node_info: NodeInfo,
    current_user: User = Depends(get_current_active_user)
):
    if node_info.user_id != current_user.username:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized for this node"
        )
        
    node_info.last_heartbeat = datetime.now()
    
    # Update compute contribution
    token_system.record_compute_contribution(
        node_info.user_id,
        node_id,
        datetime.now(),
        datetime.now(),
        node_info.resources
    )
    
    # Update node metrics
    metrics_collector.record_node_metrics(
        node_id,
        {
            "resources": node_info.resources,
            "status": node_info.status
        }
    )
    
    return {"status": "ok"}

# Job Management
@app.post("/jobs/submit")
async def submit_job(
    job: JobSubmission,
    current_user: User = Depends(get_current_active_user)
):
    # Check if user has enough tokens
    estimated_cost = token_system.estimate_compute_cost(
        job.requirements,
        24.0  # Assume 24-hour maximum duration for now
    )
    
    if not token_system.check_compute_access(current_user.username, estimated_cost):
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Insufficient tokens for compute access"
        )
    
    # Submit job
    job_id = str(uuid.uuid4())
    job_config = {
        "job_id": job_id,
        "user_id": current_user.username,
        **job.dict()
    }
    
    result = await job_scheduler.submit_job(job_config)
    if result:
        # Reserve tokens for the job
        token_system.spend_tokens(current_user.username, estimated_cost)
        return {"job_id": job_id, "status": "submitted"}
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
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

# Monitoring endpoints
@app.get("/metrics/training/{job_id}")
async def get_training_metrics(
    job_id: str,
    current_user: User = Depends(get_current_active_user)
):
    job_status = job_scheduler.get_job_status(job_id)
    if job_status["status"] == "not_found":
        raise HTTPException(status_code=404, detail="Job not found")
        
    if job_status.get("user_id") != current_user.username:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized for this job"
        )
        
    return metrics_collector.get_training_metrics(job_id)

@app.get("/metrics/node/{node_id}")
async def get_node_metrics(
    node_id: str,
    current_user: User = Depends(get_current_active_user)
):
    return metrics_collector.get_node_metrics(node_id)

@app.get("/metrics/system")
async def get_system_metrics(
    current_user: User = Depends(get_current_active_user)
):
    if not check_permissions(current_user, "admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return metrics_collector.get_system_summary()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        workers=API_CONFIG["workers"]
    )
