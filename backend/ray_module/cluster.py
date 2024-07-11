import ray
from typing import Dict, Optional
import psutil
import GPUtil
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClusterManager:
    def __init__(self):
        self.node_cleanup_threshold = timedelta(minutes=5)

    def initialize_ray(self, redis_address: Optional[str] = None):
        """Initialize Ray cluster with optional redis address for existing cluster"""
        try:
            if not ray.is_initialized():
                if redis_address:
                    ray.init(address=redis_address, namespace="distlm")
                else:
                    ray.init(namespace="distlm")
            logger.info("Ray initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Ray: {str(e)}")
            return False

    def get_node_resources(self) -> Dict[str, float]:
        """Get available resources on the current node"""
        resources = {
            "cpu_cores": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(),
            "memory_total": psutil.virtual_memory().total / (1024 ** 3),  # GB
            "memory_available": psutil.virtual_memory().available / (1024 ** 3),  # GB
        }

        # Add GPU information if available
        try:
            gpus = GPUtil.getGPUs()
            for idx, gpu in enumerate(gpus):
                resources[f"gpu_{idx}_memory_total"] = gpu.memoryTotal / 1024  # GB
                resources[f"gpu_{idx}_memory_free"] = gpu.memoryFree / 1024  # GB
                resources[f"gpu_{idx}_utilization"] = gpu.load * 100  # percentage
        except Exception as e:
            logger.warning(f"No GPU information available: {str(e)}")

        return resources

    @ray.remote
    def monitor_node_health(self, node_id: str):
        """Remote task to monitor node health"""
        try:
            resources = self.get_node_resources()
            return {
                "node_id": node_id,
                "status": "healthy",
                "timestamp": datetime.now(),
                "resources": resources
            }
        except Exception as e:
            return {
                "node_id": node_id,
                "status": "unhealthy",
                "timestamp": datetime.now(),
                "error": str(e)
            }

    def check_node_timeout(self, last_heartbeat: datetime) -> bool:
        """Check if node has timed out based on last heartbeat"""
        if not last_heartbeat:
            return True
        return datetime.now() - last_heartbeat > self.node_cleanup_threshold

    def get_cluster_status(self):
        """Get overall cluster status"""
        try:
            return {
                "cluster_nodes": len(ray.nodes()),
                "total_resources": ray.cluster_resources(),
                "available_resources": ray.available_resources()
            }
        except Exception as e:
            logger.error(f"Failed to get cluster status: {str(e)}")
            return None

    def cleanup_dead_nodes(self, nodes: Dict):
        """Clean up nodes that haven't sent heartbeat recently"""
        current_time = datetime.now()
        dead_nodes = []
        
        for node_id, node_info in nodes.items():
            if self.check_node_timeout(node_info.last_heartbeat):
                dead_nodes.append(node_id)
                
        return dead_nodes 