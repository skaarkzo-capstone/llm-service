from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from util import prompt
import torch
import json
from datetime import datetime


# To make sure it utilizes the GPU if CUDA is installed
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Fetches and downloads the model if not already done so
model_id = "meta-llama/Llama-3.1-8B-Instruct"

# Define the quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Sets up the tokenizer (Think of it as an encoder and decoder of requests and responses to and from the LLM)
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Initialize the model
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", quantization_config=quantization_config, attn_implementation="flash_attention_2")
# Mount the entire model onto GPU
model.to(device)

# Function to evaluate companies given scraped data
def evaluate(content):
    system_role = (prompt.system_prompt)
    
    # Prepare the input as before
    chat = [
        {"role": "system", "content": system_role},
        {"role": "user", "content": content}
    ]

    # 2: Apply the chat template
    formatted_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    print("Formatted chat:\n", formatted_chat)

    # 3: Tokenize the chat
    inputs = tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False)

    print(len(inputs["input_ids"][0])) 

    # Mount the tokenized inputs on GPU
    inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
    print("Tokenized inputs:\n", inputs)

    # 4: Generate text from the model
    outputs = model.generate(**inputs, max_new_tokens=1000)
    print("Generated tokens:\n", outputs)
    print(len(outputs)) 

    # 5: Decode the output back to a string
    decoded_output = tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
    print("Decoded output:\n", decoded_output)
    print(type(decoded_output))

    data = json.loads(decoded_output)

    data["date"] = datetime.today().strftime('%Y-%m-%d')
    data["score"] = int(data["scores"]["green"]) + int(data["scores"]["decarbonization"]) + int(data["scores"]["social"])
    data["compliance"] = data["score"] >= 5
    
    print(data)

    return data


# Function to summarize any given content
def summarize(content):
    system_role = (
        """Summarize the given input and send the response back. Nothing more, nothing less."""
    )
    
    # Prepare the input as before
    chat = [
        {"role": "system", "content": system_role},
        {"role": "user", "content": content}
    ]

    # 2: Apply the chat template
    formatted_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    print("Formatted chat:\n", formatted_chat)

    # 3: Tokenize the chat
    inputs = tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False)

    print(len(inputs["input_ids"][0])) 

    # Move the tokenized inputs to the same device the model is on (GPU/CPU)
    inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
    print("Tokenized inputs:\n", inputs)

    # 4: Generate text from the model
    outputs = model.generate(**inputs, max_new_tokens=1000, temperature=0.1)
    print("Generated tokens:\n", outputs)

    # 5: Decode the output back to a string
    decoded_output = tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
    print("Decoded output:\n", decoded_output)
    print(type(decoded_output))

    return decoded_outputdef chunk_text(text, chunk_size, overlap):
    """
    Splits text into word-based chunks of size `chunk_size`.
    overlap helps maintain continuity between chunks.
    """
    words = text.split()
    chunks = []
    start = 0

    # Loop through and 'chunkify' for every chunksize words
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += (chunk_size - overlap if chunk_size - overlap > 0 else chunk_size)

    return chunks
