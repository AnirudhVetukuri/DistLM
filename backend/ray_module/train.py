import ray
import torch
import torch.optim as optim
from ray_module.data_loader import load_data
from ray_module.model import load_model
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

@ray.remote
def train_model(model_file=None, dataset_file=None, epochs=1):
    # Use default model and tokenizer for testing if no model file is provided
    if model_file is None:
        model_name = "meta-llama/Meta-Llama-3-8B"
        token = None # Add HuggingFace Token here (Write token only)
        
        if not token:
            raise ValueError("Hugging Face token not found in environment variables.")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
        model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token)
    else:
        model, tokenizer = load_model(model_file)

    # Use default dataset for testing if no dataset file is provided
    if dataset_file is None:
        from datasets import load_dataset
        dataset = load_dataset('ag_news', split='train[:1%]')
        texts = dataset['text']
    else:
        texts = load_data(dataset_file)
    
    # Tokenize the text data
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f'Train Epoch: {epoch} Loss: {loss.item()}')
        
    return "Training complete"
