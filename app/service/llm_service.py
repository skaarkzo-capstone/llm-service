from fastapi import HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import json

# To make sure it utilizes the GPU if CUDA is installed
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Fetches and downloads the model if not already done so
model_id = "meta-llama/Llama-3.1-8B-Instruct"

# Sets up the tokenizer (Think of it as an encoder and decoder of requests and responses to and from the LLM)
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Load the model
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=BitsAndBytesConfig(load_in_8bit=True))
# Ensure the model uses the GPU
#model = model.to(device)

def generate(prompt):
    try:
        # Tokenize the input text
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        # Generate an output from the LLM
        output = model.generate(input_ids, max_new_tokens=512, temperature=0.1)
        # Decode the output in human-readable form
        response_text = tokenizer.decode(output[0], skip_special_tokens=True)

        return {"response":response_text}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

def chat(content):
    system_role = (
        """You are an ESG analyst tasked with evaluating companies for alignment with sustainable finance criteria based on RBC's Sustainable Finance Framework. 
        Your task is to evaluate each company’s activities and provide a sustainability score between 0 and 100 based on their alignment with the 
        following categories: Green Activities, Decarbonization Activities, and Social Activities."""

        """Scoring Guidelines:"""

        """
        90-100: The company demonstrates exemplary alignment with all three categories and contributes meaningfully to sustainable development.
        70-89: The company aligns well with at least two categories and meets the minimum standards in the third.
        50-69: The company aligns with at least one category and demonstrates partial progress in others.
        0-49: The company lacks significant alignment or operates in exclusionary industries.
        """
        
        
        """Activities to Evaluate:"""

        """
        Green Activities: Renewable energy, energy efficiency, pollution prevention, sustainable resource management, clean transportation, green buildings, climate adaptation, and circular economy initiatives.
        Decarbonization Activities: Carbon capture, electrification of industrial processes, low-carbon fuels, and methane reduction.
        Social Activities: Essential services, affordable housing, infrastructure for underserved communities, and socioeconomic advancement programs."""
        
        
        """Reasoning:"""

        """Green: Describe how the company aligns (or does not align) with green activities.
        Decarbonization: Describe how the company aligns (or does not align) with decarbonization activities.
        Social: Describe how the company aligns (or does not align) with social activities."""
        
        """
        Instructions:

        Assess the company’s activities based on the framework above.
        Assign a score based on their overall alignment across the categories.
        Explain your reasoning under the green, decarbonization, and social keys in the JSON output.
        """
        """ONLY OUTPUT THE JSON AND NOTHING ELSE:"""
        """The output must strictly adhere to the JSON format described below:"""

        """
        {
        "name": "str",
        "date": "datetime",
        "score": "int",
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
    outputs = model.generate(**inputs, max_new_tokens=1000, temperature=0.1)
    print("Generated tokens:\n", outputs)

    # 5: Decode the output back to a string
    decoded_output = tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
    print("Decoded output:\n", decoded_output)
    print(type(decoded_output))

    json_object = json.loads(decoded_output)
    print(json_object)

    return json_object