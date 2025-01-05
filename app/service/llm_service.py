from fastapi import HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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
# Load the model
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", quantization_config=quantization_config, attn_implementation="flash_attention_2")
# Ensure the entire model uses the GPU
model.to(device)

def chat(content):
    system_role = (
        """You are an ESG analyst tasked with evaluating companies for alignment with sustainable finance criteria based on RBC's Sustainable Finance Framework. 
        Your task is to evaluate each company’s activities and provide a sustainability score based on their alignment with the 
        following categories: Green Activities, Decarbonization Activities, and Social Activities."""

        """Activities to Evaluate:"""

        
        """Look if company has any of the following Green Activities: Renewable energy, energy efficiency, pollution prevention, sustainable resource management, clean transportation, green buildings, climate adaptation, and circular economy initiatives."""
        """Look if company has any of the following Decarbonization Activities: Carbon capture, electrification of industrial processes, low-carbon fuels, and methane reduction."""
        """Look if company has any of the following Social Activities: Essential services, affordable housing, infrastructure for underserved communities, and socioeconomic advancement programs."""
        
        """Scoring Guidelines:"""

        """
        For each section (Green, Decarbonization, Social):
        - If company has 1 activity: Score of 1
        - If company has 2 activities: Score of 2
        - If company has more than 2 activities: Score of 3 (Green Activities section scores up to 4)
        """

        """Reasoning:"""

        """
        Green: Describe exactly which green activities the company aligns with and explain.
        Decarbonization: Describe exactly which decarbonization activities the company aligns with and explain.
        Social: Describe exactly which social activities the company aligns with and explain."""

        """
        Instructions:

        Assess the company’s activities based on the framework above.
        Assign a score for each section based on the number of activities the company aligns with.
        Provide reasoning for the scores under the green, decarbonization, and social keys in the JSON output.
        """

        """ONLY OUTPUT THE JSON AND NOTHING ELSE:"""
        """The output must strictly adhere to the JSON format described below:"""

        """
        {
        "name": "str",
        "date": "datetime",
        "scores": {
            "green": "int",
            "decarbonization": "int",
            "social": "int"
        },
        "reasoning": {
            "green": "str",
            "decarbonization": "str",
            "social": "str"
            }
        }
        """
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

    return decoded_output