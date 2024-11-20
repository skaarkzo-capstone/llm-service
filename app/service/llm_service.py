from fastapi import HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# To make sure it utilizes the GPU if CUDA is installed
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Fetches and downloads the model if not already done so
model_id = "meta-llama/Llama-3.1-8B-Instruct"

# Sets up the tokenizer (Think of it as an encoder and decoder of requests and responses to and from the LLM)
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Load the model
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
# Ensure the model uses the GPU
model = model.to(device)

def generate(prompt):
    try:
        # Tokenize the input text
        input_ids = tokenizer.encode(prompt.input_text, return_tensors="pt").to(device)
        # Generate an output from the LLM
        output = model.generate(input_ids, max_new_tokens=512, temperature=0.1)
        # Decode the output in human-readable form
        response_text = tokenizer.decode(output[0], skip_special_tokens=True)

        return {"response":response_text}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

def chat(content):
    system_role = (
        "You are a financial analyst specializing in sustainability and environmentally friendly investments. "
        "Evaluate the company data against the following eligibility criteria: "
        "1. Pollution prevention and control: Projects involving collection, treatment, or recycling of emissions, waste, or contaminated soil. "
        "2. Sustainable resource management: Activities contributing to sustainable agriculture, forestry, or biodiversity conservation. "
        "3. Clean transportation: Development or maintenance of low-emission transportation systems. "
        "4. Water management: Projects improving water treatment, recycling, or conservation. "
        "5. Climate resilience: Adaptation projects reducing vulnerability to climate impacts. "
        "6. Circular economy: Activities substituting virgin raw materials with recycled ones or promoting reuse. "
        "Assess whether the provided company initiatives align with the UN Sustainable Development Goals (SDGs)."
    )
    # Prepare the input as before
    chat = [
        {"role": "system", "content": system_role},
        {"role": "user", "content": content.input_text}
    ]

    # 2: Apply the chat template
    formatted_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    print("Formatted chat:\n", formatted_chat)

    # 3: Tokenize the chat
    inputs = tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False)

    # Move the tokenized inputs to the same device the model is on (GPU/CPU)
    inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
    print("Tokenized inputs:\n", inputs)

    # 4: Generate text from the model
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1)
    print("Generated tokens:\n", outputs)

    # 5: Decode the output back to a string
    decoded_output = tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
    print("Decoded output:\n", decoded_output)

    return True