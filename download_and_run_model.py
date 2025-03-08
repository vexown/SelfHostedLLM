# Meta Llama 3 8B Instruct
# Meta Llama 3 is licensed under the Meta Llama 3 Community License, 
# Copyright © Meta Platforms, Inc. All Rights Reserved.
#
# This script downloads and runs the Meta Llama 3 8B Instruct model.
# Please ensure you have read and agree to the license and Acceptable Use Policy.
# For more information, visit https://llama.meta.com/llama3

from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
import torch
import os

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
local_model_path = "./llama-3-8b-instruct"

# Read the Hugging Face token from the file
try:
    with open("huggingface_token.txt", "r") as f:
        your_token = f.read().strip()
except FileNotFoundError:
    print("Error: huggingface_token.txt file not found.")
    exit(1)  # Exit with an error code

# Download the model if it doesn't exist locally
if not os.path.exists(local_model_path):
    snapshot_download(repo_id=model_name, local_dir=local_model_path, token=your_token)

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(local_model_path, torch_dtype=torch.float16).to("cuda") #if you have cuda.

# Example inference
input_text = "What is the capital of France?"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_length=200)
generated_text = tokenizer.decode(outputs[0])

print(generated_text)