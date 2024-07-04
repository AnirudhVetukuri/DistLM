import os
from transformers import LlamaForCausalLM, LlamaTokenizer

def load_model(model_file):
    model_path = os.path.join("uploads", "models", model_file)
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(model_path)
    
    return model, tokenizer
