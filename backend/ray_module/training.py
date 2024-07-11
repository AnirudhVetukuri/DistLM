import ray
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@ray.remote(num_gpus=1)  # Specify GPU requirement
class DistributedTrainer:
    def __init__(self, model_config: Dict[str, Any]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._initialize_model(model_config)
        self.model.to(self.device)
        
    def _initialize_model(self, model_config: Dict[str, Any]) -> nn.Module:
        """Initialize model based on configuration"""
        # This is a placeholder - extend with actual model architectures
        model_type = model_config.get("type", "simple_transformer")
        if model_type == "simple_transformer":
            from transformers import AutoModelForCausalLM
            return AutoModelForCausalLM.from_pretrained(
                model_config.get("model_name", "gpt2"),
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def train_batch(self, batch_data: Dict[str, torch.Tensor], 
                   optimizer_config: Dict[str, Any]) -> Dict[str, float]:
        """Train on a single batch of data"""
        try:
            # Move data to device
            input_ids = batch_data["input_ids"].to(self.device)
            attention_mask = batch_data.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            labels = batch_data.get("labels", input_ids).to(self.device)

            # Initialize optimizer
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=optimizer_config.get("learning_rate", 1e-4),
                weight_decay=optimizer_config.get("weight_decay", 0.01)
            )

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            return {
                "loss": loss.item(),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error in training batch: {str(e)}")
            return {"error": str(e)}

    def get_model_state(self) -> Dict[str, torch.Tensor]:
        """Get model state for synchronization"""
        return {
            name: param.cpu() 
            for name, param in self.model.state_dict().items()
        }

    def set_model_state(self, state_dict: Dict[str, torch.Tensor]):
        """Update model with new state"""
        self.model.load_state_dict(state_dict)

    def validate(self, val_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Run validation on a batch of data"""
        self.model.eval()
        with torch.no_grad():
            try:
                input_ids = val_data["input_ids"].to(self.device)
                attention_mask = val_data.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                labels = val_data.get("labels", input_ids).to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                return {
                    "val_loss": outputs.loss.item(),
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                logger.error(f"Error in validation: {str(e)}")
                return {"error": str(e)}
        
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'timestamp': datetime.now().isoformat()
            }, path)
            return True
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            return False

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        try:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            return True
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            return False 