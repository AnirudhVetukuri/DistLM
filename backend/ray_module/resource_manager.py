from typing import Dict, List, Optional
import ray
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResourceManager:
    def __init__(self):
        self.allocated_resources = {}  # job_id -> resources
        self.resource_history = {}     # node_id -> [(timestamp, usage)]
        self.history_limit = 1000      # Keep last 1000 records per node

    def check_resource_availability(self, requirements: Dict[str, float], nodes: Dict) -> List[str]:
        """
        Check which nodes can satisfy the resource requirements
        Returns list of suitable node IDs
        """
        suitable_nodes = []
        
        for node_id, node_info in nodes.items():
            if node_info.status != "available":
                continue
                
            resources = node_info.resources
            can_satisfy = True
            
            for resource, required in requirements.items():
                available = resources.get(resource, 0)
                if available < required:
                    can_satisfy = False
                    break
                    
            if can_satisfy:
                suitable_nodes.append(node_id)
                
        return suitable_nodes

    def allocate_resources(self, job_id: str, requirements: Dict[str, float], node_id: str) -> bool:
        """
        Attempt to allocate resources on a specific node for a job
        Returns True if allocation successful
        """
        try:
            # Get current node resources from Ray
            node_resources = ray.available_resources()
            
            # Check if resources are available
            for resource, required in requirements.items():
                if node_resources.get(resource, 0) < required:
                    logger.warning(f"Insufficient {resource} on node {node_id}")
                    return False
            
            # Record allocation
            self.allocated_resources[job_id] = {
                "node_id": node_id,
                "resources": requirements,
                "allocated_at": datetime.now()
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to allocate resources: {str(e)}")
            return False

    def release_resources(self, job_id: str) -> bool:
        """
        Release resources allocated to a job
        Returns True if release successful
        """
        try:
            if job_id in self.allocated_resources:
                allocation = self.allocated_resources.pop(job_id)
                logger.info(f"Released resources for job {job_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to release resources: {str(e)}")
            return False

    def track_resource_usage(self, node_id: str, usage: Dict[str, float]):
        """Record resource usage history for a node"""
        if node_id not in self.resource_history:
            self.resource_history[node_id] = []
            
        history = self.resource_history[node_id]
        history.append((datetime.now(), usage))
        
        # Maintain history limit
        if len(history) > self.history_limit:
            history.pop(0)

    def get_resource_usage_history(self, node_id: str) -> List:
        """Get resource usage history for a node"""
        return self.resource_history.get(node_id, [])

    def get_current_allocations(self) -> Dict:
        """Get current resource allocations"""
        return self.allocated_resources

    def calculate_node_load(self, node_id: str) -> Dict[str, float]:
        """Calculate current load percentages for a node"""
        if node_id not in self.resource_history:
            return {}
            
        history = self.resource_history[node_id]
        if not history:
            return {}
            
        # Get most recent usage
        _, latest_usage = history[-1]
        
        # Calculate percentages
        load = {}
        for resource, value in latest_usage.items():
            if resource.endswith("_total"):
                resource_name = resource[:-6]  # Remove _total suffix
                if f"{resource_name}_available" in latest_usage:
                    total = value
                    available = latest_usage[f"{resource_name}_available"]
                    used_percent = ((total - available) / total) * 100
                    load[f"{resource_name}_utilization"] = used_percent
                    
        return load 