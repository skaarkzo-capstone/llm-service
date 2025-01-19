from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from app.util import prompt
import torch
import json
from datetime import datetime


# To make sure it utilizes the GPU if CUDA is installed
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Fetches and downloads the model if not already done so
model_id = "meta-llama/Llama-3.2-3B-Instruct"

# Define the quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16
)

# Sets up the tokenizer (Think of it as an encoder and decoder of requests and responses to and from the LLM)
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Initialize the model
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto", attn_implementation="flash_attention_2")
# Mount the entire model onto GPU
#model.to(device)

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

# Function that summarizes using the model
def summarize_with_generate(text, max_new_tokens=150):
    """
    Summarize using HF generate().
    """

    prompt = f"""You are a strict filter. 
    If the following text is related to sustainability, produce a concise summary of the sustainability-related aspects only.
    Do not output phrases like ‘I will output nothing.’ If there is no sustainability text, your entire output should be blank—i.e., an empty string.
    If not related to sustainability, produce absolutely nothing (no disclaimers, no placeholders).

    Text:
    {text}

    Your output:
    """
    
    # Tokenize the inputs
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate the summary
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0
    )
    
    input_length = inputs["input_ids"].shape[1]
    summary = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    return summary.strip()

# Splits a given input into defined sized chunks
def chunk_text(text, chunk_size, overlap):
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

# Initiate the summarization process
def summarize(input, chunk_size=1024, overlap=0, max_new_tokens=100):
    """
    1) Chunk the large text based on the chunk size
    2) Summarize each chunk individually and output length based on the max tokens field
    3) Combine those summaries into one string once all finished
    """
    
    text = json.dumps(input, indent=2) # Prettify json into text
    print(text)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    partial_summaries = []
    
    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i+1}/{len(chunks)}...")
        summary = summarize_with_generate(chunk, max_new_tokens=max_new_tokens)
        partial_summaries.append(summary)
    
    combined_summary = "\n".join(partial_summaries)
    return combined_summary
    