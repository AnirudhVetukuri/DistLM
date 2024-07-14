import ray
from typing import Dict, Any, Iterator, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset as HFDataset
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@ray.remote
class DistributedDataHandler:
    def __init__(self, data_config: Dict[str, Any]):
        self.data_config = data_config
        self.tokenizer = self._initialize_tokenizer()
        self.dataset = self._load_dataset()

    def _initialize_tokenizer(self):
        """Initialize tokenizer based on model configuration"""
        try:
            model_name = self.data_config.get("model_name", "gpt2")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Ensure the tokenizer has padding token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        except Exception as e:
            logger.error(f"Error initializing tokenizer: {str(e)}")
            raise

    def _load_dataset(self) -> HFDataset:
        """Load dataset from file or HuggingFace datasets"""
        try:
            data_source = self.data_config.get("data_source", "file")
            
            if data_source == "huggingface":
                # Load from HuggingFace datasets
                dataset_name = self.data_config["dataset_name"]
                return load_dataset(dataset_name)
            
            elif data_source == "file":
                # Load from local file
                file_path = self.data_config["file_path"]
                return self._load_from_file(file_path)
            
            else:
                raise ValueError(f"Unsupported data source: {data_source}")
                
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise

    def _load_from_file(self, file_path: str) -> HFDataset:
        """Load dataset from local file"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        if path.suffix == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
            return HFDataset.from_dict(data)
            
        elif path.suffix == '.txt':
            with open(path, 'r') as f:
                texts = [line.strip() for line in f if line.strip()]
            return HFDataset.from_dict({"text": texts})
            
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def preprocess_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Preprocess a batch of data"""
        try:
            # Get text from batch
            if isinstance(batch, dict):
                text = batch.get("text", batch.get("content", ""))
            else:
                text = batch
                
            # Tokenize
            tokenized = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=self.data_config.get("max_length", 512),
                return_tensors="pt"
            )
            
            # Prepare for language modeling (shift labels for next token prediction)
            labels = tokenized["input_ids"].clone()
            
            return {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": labels
            }
            
        except Exception as e:
            logger.error(f"Error preprocessing batch: {str(e)}")
            raise

    def get_data_loader(self, split: str = "train") -> DataLoader:
        """Get DataLoader for specified split"""
        try:
            # Get dataset split
            dataset_split = self.dataset[split]
            
            # Create custom Dataset
            class TextDataset(Dataset):
                def __init__(self, data, preprocessor):
                    self.data = data
                    self.preprocessor = preprocessor
                    
                def __len__(self):
                    return len(self.data)
                    
                def __getitem__(self, idx):
                    item = self.data[idx]
                    return self.preprocessor(item)
            
            # Create DataLoader
            dataset = TextDataset(dataset_split, self.preprocess_batch)
            
            return DataLoader(
                dataset,
                batch_size=self.data_config.get("batch_size", 16),
                shuffle=(split == "train"),
                num_workers=self.data_config.get("num_workers", 2)
            )
            
        except Exception as e:
            logger.error(f"Error creating DataLoader: {str(e)}")
            raise

    def get_vocab_size(self) -> int:
        """Get vocabulary size from tokenizer"""
        return len(self.tokenizer)

    def encode_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Encode text for inference"""
        return self.preprocess_batch({"text": text})

    def decode_ids(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs back to text"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    @property
    def pad_token_id(self) -> int:
        """Get pad token ID"""
        return self.tokenizer.pad_token_id 