from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path
import psutil
import GPUtil
from ..config import LOG_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsCollector:
    def __init__(self):
        self.metrics_history: Dict[str, List[Dict]] = {}
        self.node_metrics: Dict[str, List[Dict]] = {}
        self.training_metrics: Dict[str, List[Dict]] = {}
        self.history_limit = 1000

    def record_training_metrics(self, job_id: str, metrics: Dict[str, float]):
        """Record training metrics for a job"""
        if job_id not in self.training_metrics:
            self.training_metrics[job_id] = []
            
        metrics["timestamp"] = datetime.now().isoformat()
        self.training_metrics[job_id].append(metrics)
        
        if len(self.training_metrics[job_id]) > self.history_limit:
            self.training_metrics[job_id].pop(0)
            
        self._save_metrics(job_id, metrics, "training")

    def record_node_metrics(self, node_id: str, metrics: Dict[str, Any]):
        """Record node metrics"""
        if node_id not in self.node_metrics:
            self.node_metrics[node_id] = []
            
        metrics["timestamp"] = datetime.now().isoformat()
        self.node_metrics[node_id].append(metrics)
        
        if len(self.node_metrics[node_id]) > self.history_limit:
            self.node_metrics[node_id].pop(0)
            
        self._save_metrics(node_id, metrics, "node")

    def get_training_metrics(self, job_id: str, 
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None) -> List[Dict]:
        """Get training metrics for a job within time range"""
        if job_id not in self.training_metrics:
            return []
            
        metrics = self.training_metrics[job_id]
        
        if start_time or end_time:
            filtered_metrics = []
            for metric in metrics:
                metric_time = datetime.fromisoformat(metric["timestamp"])
                if start_time and metric_time < start_time:
                    continue
                if end_time and metric_time > end_time:
                    continue
                filtered_metrics.append(metric)
            return filtered_metrics
            
        return metrics

    def get_node_metrics(self, node_id: str,
                        start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None) -> List[Dict]:
        """Get node metrics within time range"""
        if node_id not in self.node_metrics:
            return []
            
        metrics = self.node_metrics[node_id]
        
        if start_time or end_time:
            filtered_metrics = []
            for metric in metrics:
                metric_time = datetime.fromisoformat(metric["timestamp"])
                if start_time and metric_time < start_time:
                    continue
                if end_time and metric_time > end_time:
                    continue
                filtered_metrics.append(metric)
            return filtered_metrics
            
        return metrics

    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""
        metrics = {
            "cpu": {
                "percent": psutil.cpu_percent(interval=1),
                "count": psutil.cpu_count(),
                "frequency": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            },
            "memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "percent": psutil.virtual_memory().percent
            },
            "disk": {
                "total": psutil.disk_usage("/").total,
                "used": psutil.disk_usage("/").used,
                "free": psutil.disk_usage("/").free,
                "percent": psutil.disk_usage("/").percent
            },
            "network": {
                "bytes_sent": psutil.net_io_counters().bytes_sent,
                "bytes_recv": psutil.net_io_counters().bytes_recv
            },
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            gpus = GPUtil.getGPUs()
            metrics["gpu"] = []
            for gpu in gpus:
                metrics["gpu"].append({
                    "id": gpu.id,
                    "name": gpu.name,
                    "load": gpu.load,
                    "memory": {
                        "total": gpu.memoryTotal,
                        "used": gpu.memoryUsed,
                        "free": gpu.memoryFree
                    },
                    "temperature": gpu.temperature
                })
        except Exception as e:
            logger.warning(f"Could not collect GPU metrics: {str(e)}")
            
        return metrics

    def _save_metrics(self, id: str, metrics: Dict, metric_type: str):
        """Save metrics to file"""
        try:
            metrics_dir = LOG_DIR / "metrics" / metric_type
            metrics_dir.mkdir(parents=True, exist_ok=True)
            
            metrics_file = metrics_dir / f"{id}.jsonl"
            
            with open(metrics_file, "a") as f:
                f.write(json.dumps(metrics) + "\n")
                
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")

    def get_system_summary(self) -> Dict[str, Any]:
        """Get system-wide metrics summary"""
        summary = {
            "nodes": {
                "total": len(self.node_metrics),
                "active": sum(1 for metrics in self.node_metrics.values()
                            if (datetime.now() - datetime.fromisoformat(metrics[-1]["timestamp"]))
                            < timedelta(minutes=5))
            },
            "training": {
                "total_jobs": len(self.training_metrics),
                "active_jobs": sum(1 for metrics in self.training_metrics.values()
                                 if (datetime.now() - datetime.fromisoformat(metrics[-1]["timestamp"]))
                                 < timedelta(minutes=5))
            },
            "system": self.collect_system_metrics()
        }
        
        return summary

metrics_collector = MetricsCollector() 