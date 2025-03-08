# Meta Llama 3.2 1B Instruct
# Optimized for low VRAM (6GB GPU)
# Uses 8-bit quantization and automatic device mapping
# Includes fallback to CPU if GPU inference fails

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import snapshot_download
import torch
import os
import time
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run Llama 3.2 1B model locally')
parser.add_argument('--prompt', type=str, default="What is the capital of France?", 
                    help='Text prompt for the model')
args = parser.parse_args()

# Ensure required libraries are installed
try:
    import bitsandbytes  # Required for quantization
except ImportError:
    print("Installing bitsandbytes...")
    os.system("pip install bitsandbytes")

# Clear GPU cache before loading the model to free up memory
torch.cuda.empty_cache()

# Set the model and local path
model_name = "meta-llama/Llama-3.2-1B-Instruct"
local_model_path = "./llama-3.2-1b-instruct"

# Read the Hugging Face token from file
try:
    with open("huggingface_token.txt", "r") as f:
        your_token = f.read().strip()
except FileNotFoundError:
    print("Error: huggingface_token.txt file not found.")
    exit(1)

# Download the model if it doesn't exist locally
if not os.path.exists(local_model_path):
    print(f"Downloading model {model_name}...")
    snapshot_download(repo_id=model_name, local_dir=local_model_path, token=your_token)
    print("Download complete.")
else:
    print(f"Model already exists at {local_model_path}")

# Set up 8-bit quantization config for better stability
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,       # 8-bit quantization for stability
    llm_int8_enable_fp32_cpu_offload=True  # Allow CPU offloading
)

# Load the tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is properly set

try:
    # Load the model with stable configuration
    print("Loading model...")
    load_start_time = time.time()
    
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        quantization_config=bnb_config,  # Use 8-bit quantization
        device_map="auto",               # Distribute across available hardware
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    load_time = time.time() - load_start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # Display memory information
    if torch.cuda.is_available():
        print(f"Model loaded on GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    # Get user prompt or use default
    input_text = args.prompt
    print(f"\nPrompt: {input_text}")
    
    # Prepare input
    inputs = tokenizer(input_text, return_tensors="pt").to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate with timing
    print("Generating response...")
    gen_start_time = time.time()
    
    # Use conservative generation parameters
    outputs = model.generate(
        **inputs,
        max_length=100,
        do_sample=False,  # Greedy decoding
        temperature=None,  # Remove conflicting parameter
        top_p=None,        # Remove conflicting parameter
        pad_token_id=tokenizer.eos_token_id
    )
    
    gen_time = time.time() - gen_start_time
    
    # Process output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nResponse: {generated_text}")
    print(f"Generation completed in {gen_time:.2f} seconds")

except RuntimeError as e:
    print(f"GPU inference failed with error: {e}")
    print("\nFalling back to CPU-only inference with minimal settings...")
    
    # Try CPU-only as ultimate fallback
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable CUDA
    torch.cuda.empty_cache()
    
    cpu_start_time = time.time()
    
    # Load model in CPU-only mode with minimal settings
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # Get user prompt or use default
    input_text = args.prompt
    print(f"\nPrompt: {input_text}")
    
    inputs = tokenizer(input_text, return_tensors="pt")
    
    print("Generating response (CPU mode)...")
    cpu_gen_start = time.time()
    
    outputs = model.generate(
        **inputs,
        max_length=50,  # Reduced for CPU performance
        do_sample=False
    )
    
    cpu_gen_time = time.time() - cpu_gen_start
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nResponse: {generated_text}")
    print(f"CPU generation completed in {cpu_gen_time:.2f} seconds")