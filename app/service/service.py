from fastapi import HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# To make sure it utilizes the GPU if CUDA is installed
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Fetches and downloads the model if not already done so
model_id = "meta-llama/Llama-3.1-8B"

# Sets up the tokenizer (Think of it as an encoder and decoder of requests and responses to and from the LLM)
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Load the model
model = AutoModelForCausalLM.from_pretrained(model_id)
# Ensure the model uses the GPU
model = model.to(device)

def generate(prompt):
    try:
        # Tokenize the input text
        input_ids = tokenizer.encode(prompt.input_text, return_tensors="pt").to(device)
        # Generate an output from the LLM
        output = model.generate(input_ids, max_length=50, temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=2.0, no_repeat_ngram_size=2)
        # Decode the output in human-readable form
        response_text = tokenizer.decode(output[0], skip_special_tokens=True)

        return {"response":response_text}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
