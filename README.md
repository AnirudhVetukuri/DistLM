# DistLM: Distributed Training Framework for Large Language Models

## Overview

DistLM is a distributed training framework designed for Large Language Models (LLMs) using Ray. It enables users to share compute resources and earn tokens, which can be used to access larger compute clusters for training. The system implements a token-based economy where users can contribute idle compute resources to earn tokens and spend them on training their own models.

## Features

### Core Infrastructure
- Distributed training using PyTorch and Ray
- Support for transformer models via HuggingFace
- Automatic resource allocation and scheduling
- Token-based compute sharing economy
- Node health monitoring and management

### Training Capabilities
- Multi-GPU training support
- Automatic data preprocessing and batching
- Support for custom datasets and HuggingFace datasets
- Checkpoint management and model state synchronization
- Training progress monitoring and metrics collection

### Security
- JWT-based authentication
- Role-based access control (User, Node Owner, Admin)
- Protected API endpoints
- Secure token management

### Monitoring
- Real-time training metrics
- Node health monitoring
- Resource usage tracking
- System-wide metrics collection

## System Architecture

### Backend Components

1. **API Layer** (`app/main.py`)
   - FastAPI-based REST API
   - Authentication and authorization
   - Request handling and routing
   - Error handling

2. **Training Infrastructure** (`ray_module/`)
   - `training.py`: Distributed training implementation
   - `scheduler.py`: Job scheduling and management
   - `data_handler.py`: Data loading and preprocessing
   - `resource_manager.py`: Compute resource management

3. **Security** (`auth/`)
   - `auth_handler.py`: Authentication and authorization
   - JWT token management
   - User management
   - Role-based permissions

4. **Monitoring** (`monitoring/`)
   - `metrics.py`: Metrics collection and storage
   - System monitoring
   - Training progress tracking
   - Resource usage monitoring

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/DistLM.git
   cd DistLM
   ```

2. Install dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp backend/.env.example backend/.env
   # Edit .env with your configuration
   ```

## Usage

### Starting the Backend

1. Start the FastAPI server:
   ```bash
   cd backend
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
   ```

### User Management

1. Register a new user (requires admin):
   ```bash
   curl -X POST "http://localhost:8000/auth/register" \
     -H "Authorization: Bearer $TOKEN" \
     -d '{"username": "user1", "password": "pass123", "role": "node_owner"}'
   ```

2. Login:
   ```bash
   curl -X POST "http://localhost:8000/auth/token" \
     -d "username=user1&password=pass123"
   ```

### Node Management

1. Register a compute node:
   ```bash
   curl -X POST "http://localhost:8000/nodes/register" \
     -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
       "hostname": "gpu-node-1",
       "resources": {
         "gpu": 1,
         "cpu_cores": 8,
         "memory": 32
       }
     }'
   ```

### Training Jobs

1. Submit a training job:
   ```bash
   curl -X POST "http://localhost:8000/jobs/submit" \
     -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
       "model_config": {
         "type": "simple_transformer",
         "model_name": "gpt2"
       },
       "data_config": {
         "data_source": "huggingface",
         "dataset_name": "wikitext"
       },
       "requirements": {
         "gpu": 1,
         "cpu_cores": 4,
         "memory": 16
       }
     }'
   ```

## API Documentation

### Authentication Endpoints
- `POST /auth/token`: Get JWT access token
- `POST /auth/register`: Register new user (admin only)

### Node Management
- `POST /nodes/register`: Register compute node
- `GET /nodes`: List all nodes
- `POST /nodes/{node_id}/heartbeat`: Update node status

### Job Management
- `POST /jobs/submit`: Submit training job
- `GET /jobs/{job_id}`: Get job status
- `GET /jobs`: List all jobs
- `POST /jobs/{job_id}/cancel`: Cancel job

### Token Management
- `GET /tokens/balance/{user_id}`: Get token balance
- `GET /tokens/history/{user_id}`: Get compute history
- `POST /tokens/estimate`: Estimate compute cost

### Monitoring
- `GET /metrics/training/{job_id}`: Get training metrics
- `GET /metrics/node/{node_id}`: Get node metrics
- `GET /metrics/system`: Get system-wide metrics

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For questions, issues, or contributions, please open an issue on the GitHub repository or contact Abhi Vetukuri or Anir Vetukuri.
---

DistLM - Empowering distributed training for Large Language Models
