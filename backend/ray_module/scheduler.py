import ray
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
from .training import DistributedTrainer
from .resource_manager import ResourceManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JobScheduler:
    def __init__(self, resource_manager: ResourceManager):
        self.resource_manager = resource_manager
        self.active_jobs: Dict[str, Dict] = {}
        self.job_queues: Dict[str, List] = {
            "pending": [],
            "running": [],
            "completed": [],
            "failed": []
        }
        self.trainers: Dict[str, DistributedTrainer] = {}

    async def submit_job(self, job_config: Dict[str, Any]) -> str:
        """Submit a new training job"""
        try:
            job_id = job_config["job_id"]
            
            # Check resource requirements
            required_resources = job_config.get("requirements", {})
            available_nodes = self.resource_manager.check_resource_availability(
                required_resources, 
                self.resource_manager.get_current_allocations()
            )

            if not available_nodes:
                logger.warning(f"No available nodes for job {job_id}")
                self.job_queues["pending"].append(job_id)
                return job_id

            # Allocate resources
            selected_node = available_nodes[0]  # Simple selection for now
            if not self.resource_manager.allocate_resources(job_id, required_resources, selected_node):
                logger.error(f"Failed to allocate resources for job {job_id}")
                self.job_queues["failed"].append(job_id)
                return job_id

            # Initialize trainer
            trainer = DistributedTrainer.remote(job_config.get("model_config", {}))
            self.trainers[job_id] = trainer

            # Start training
            self._start_training.remote(job_id, job_config)
            
            self.active_jobs[job_id] = {
                "status": "running",
                "node_id": selected_node,
                "start_time": datetime.now(),
                "config": job_config
            }
            self.job_queues["running"].append(job_id)

            return job_id

        except Exception as e:
            logger.error(f"Error submitting job: {str(e)}")
            if job_id:
                self.job_queues["failed"].append(job_id)
            return None

    @ray.remote
    def _start_training(self, job_id: str, job_config: Dict[str, Any]):
        """Start training process for a job"""
        try:
            trainer = self.trainers[job_id]
            
            # Get training data configuration
            data_config = job_config.get("data_config", {})
            optimizer_config = job_config.get("optimizer_config", {})
            
            # Training loop
            for epoch in range(job_config.get("num_epochs", 1)):
                epoch_losses = []
                
                # Process batches
                for batch in self._get_data_batches(data_config):
                    result = ray.get(trainer.train_batch.remote(batch, optimizer_config))
                    
                    if "error" in result:
                        raise Exception(result["error"])
                        
                    epoch_losses.append(result["loss"])
                
                # Validation
                if data_config.get("validation_data"):
                    val_result = ray.get(trainer.validate.remote(data_config["validation_data"]))
                    if "error" in val_result:
                        raise Exception(val_result["error"])
                
                # Save checkpoint
                if job_config.get("checkpoint_path"):
                    trainer.save_checkpoint.remote(f"{job_config['checkpoint_path']}/epoch_{epoch}.pt")
            
            # Job completed successfully
            self._complete_job(job_id, "completed")
            
        except Exception as e:
            logger.error(f"Error in training job {job_id}: {str(e)}")
            self._complete_job(job_id, "failed", error=str(e))

    def _get_data_batches(self, data_config: Dict[str, Any]):
        """Generator for data batches"""
        # This is a placeholder - implement actual data loading logic
        # You might want to use PyTorch DataLoader or custom data loading
        return []

    def _complete_job(self, job_id: str, status: str, error: Optional[str] = None):
        """Handle job completion"""
        if job_id in self.active_jobs:
            job_info = self.active_jobs.pop(job_id)
            job_info["end_time"] = datetime.now()
            job_info["status"] = status
            if error:
                job_info["error"] = error

            # Release resources
            self.resource_manager.release_resources(job_id)
            
            # Update queues
            if job_id in self.job_queues["running"]:
                self.job_queues["running"].remove(job_id)
            self.job_queues[status].append(job_id)

            # Clean up trainer
            if job_id in self.trainers:
                del self.trainers[job_id]

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get current status of a job"""
        if job_id in self.active_jobs:
            return self.active_jobs[job_id]
        
        # Check queues
        for status, queue in self.job_queues.items():
            if job_id in queue:
                return {"status": status}
        
        return {"status": "not_found"}

    def list_jobs(self) -> Dict[str, List[str]]:
        """Get lists of all jobs by status"""
        return self.job_queues

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running or pending job"""
        try:
            if job_id in self.active_jobs:
                self._complete_job(job_id, "failed", error="Job cancelled by user")
                return True
            elif job_id in self.job_queues["pending"]:
                self.job_queues["pending"].remove(job_id)
                self.job_queues["failed"].append(job_id)
                return True
            return False
        except Exception as e:
            logger.error(f"Error cancelling job {job_id}: {str(e)}")
            return False 