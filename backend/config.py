from typing import Dict, Any
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
LOG_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, CHECKPOINT_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Ray configuration
RAY_CONFIG = {
    "address": os.getenv("RAY_ADDRESS", "auto"),
    "namespace": "distlm",
    "runtime_env": {
        "working_dir": str(BASE_DIR),
        "pip": ["torch", "transformers", "datasets"]
    }
}

# Token system configuration
TOKEN_CONFIG = {
    "storage_path": str(DATA_DIR / "token_data.json"),
    "base_rate": float(os.getenv("TOKEN_BASE_RATE", "1.0")),
    "gpu_multiplier": float(os.getenv("TOKEN_GPU_MULTIPLIER", "2.0")),
    "cpu_multiplier": float(os.getenv("TOKEN_CPU_MULTIPLIER", "0.1"))
}

# Training defaults
TRAINING_DEFAULTS = {
    "batch_size": 16,
    "max_length": 512,
    "num_epochs": 3,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "warmup_steps": 500,
    "max_grad_norm": 1.0,
    "num_workers": 2
}

# Model defaults
MODEL_DEFAULTS = {
    "type": "simple_transformer",
    "model_name": "gpt2",
    "use_fp16": True
}

# Resource limits
RESOURCE_LIMITS = {
    "max_gpu_per_job": 4,
    "max_cpu_per_job": 16,
    "max_memory_gb": 32,
    "min_gpu_memory_gb": 8
}

# API configuration
API_CONFIG = {
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", "8000")),
    "debug": bool(os.getenv("API_DEBUG", "False")),
    "workers": int(os.getenv("API_WORKERS", "4"))
}

def get_job_config(
    model_config: Dict[str, Any],
    data_config: Dict[str, Any],
    requirements: Dict[str, float]
) -> Dict[str, Any]:
    """Create a complete job configuration with defaults"""
    config = {
        "model_config": {
            **MODEL_DEFAULTS,
            **model_config
        },
        "data_config": {
            **TRAINING_DEFAULTS,
            **data_config
        },
        "requirements": {
            **{"cpu_cores": 2, "memory": 8},  # Minimum requirements
            **requirements
        }
    }
    
    # Validate resource requirements
    for resource, limit in RESOURCE_LIMITS.items():
        if resource.startswith("max_"):
            resource_name = resource[4:]  # Remove 'max_' prefix
            if resource_name in config["requirements"]:
                config["requirements"][resource_name] = min(
                    config["requirements"][resource_name],
                    limit
                )
    
    return config

def get_checkpoint_path(job_id: str) -> str:
    """Get checkpoint path for a job"""
    return str(CHECKPOINT_DIR / job_id)

def get_log_path(job_id: str) -> str:
    """Get log path for a job"""
    return str(LOG_DIR / f"{job_id}.log") 