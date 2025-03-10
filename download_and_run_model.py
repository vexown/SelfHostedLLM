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

conversation_history = ""

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run Llama 3.2 1B model locally')
parser.add_argument('--prompt', type=str, 
                    help='Text prompt for the model (optional, starts interactive mode if not provided)')
parser.add_argument('--interactive', action='store_true',
                    help='Start in interactive mode regardless of prompt argument')
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
model_name = "meta-llama/Llama-3.2-3B-Instruct"
local_model_path = "./llama-3.2-3b-instruct"

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

# Load the tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is properly set

# Function to generate response
def generate_response(input_text, model, tokenizer, device='auto', remember_history=True):
    global conversation_history
    
    print(f"\nPrompt: {input_text}")  # Only show current prompt
    
    # Track memory before prompt processing
    if torch.cuda.is_available() and device != 'cpu':
        memory_before = torch.cuda.memory_allocated() / 1024**2
        print(f"\nGPU Memory before processing: {memory_before:.2f} MB")
    
    if remember_history:
        # Append the new input to the conversation history with clear roles
        if conversation_history:
            # Add user message with clear attribution
            conversation_history += f"\nHuman: {input_text}"
            
            # Format prompt with clear roles and instructions
            full_prompt = (
                "You are an AI assistant with expertise in electronics engineering. "
                "Below is a conversation history. Respond ONLY to the most recent message "
                "from the Human, without repeating or extending the conversation history." 
                "Do NOT make up additional Human messages, respond only the the provided prompt."
                "Make your answers concise. \n\n"
                f"{conversation_history}\n\nAssistant:"
            )
        else:
            # First message in conversation
            conversation_history = f"Human: {input_text}"
            full_prompt = f"You are an AI assistant with expertise in electronics engineering. Respond to: {input_text}\n\nAssistant:"
            
        # Calculate tokens in conversation history (keep this)
        history_tokens = len(tokenizer.encode(conversation_history))
        print(f"Conversation history length: {history_tokens} tokens")
    else:
        # Original single-turn behavior
        full_prompt = "You are an AI assistant with expertise in electronics engineering. Respond directly to: " + input_text + "\n\nAssistant:"
    
    # Prepare input
    if device == 'cpu':
        inputs = tokenizer(full_prompt, return_tensors="pt")
    else:
        inputs = tokenizer(full_prompt, return_tensors="pt").to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Track memory after preparing input
    if torch.cuda.is_available() and device != 'cpu':
        memory_after_tokenizing = torch.cuda.memory_allocated() / 1024**2
        print(f"GPU Memory after tokenizing: {memory_after_tokenizing:.2f} MB (Δ +{memory_after_tokenizing - memory_before:.2f} MB)")
    
    # Generate with timing
    print("Generating response...")
    gen_start_time = time.time()
    
    # Generation parameters
    if device == 'cpu':
        outputs = model.generate(
            **inputs,
            max_length=50,  # Reduced for CPU performance
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    else:
        outputs = model.generate(
            **inputs,             # Input prompt
            max_new_tokens=200,   # Maximum number of tokens to generate
            do_sample=True,       # Enable sampling (or use greedy decoding for deterministic output by setting to False)
            temperature=0.2,      # Temperature (higher means more random, lower means more deterministic) - set to None if greedy decoding is enabled
            top_p=0.9,            # Nucleus sampling (top_p = 0.9 means 90% of the cumulative probability mass is considered). 
                                  # In simple terms, it sets a threshold to avoid unlikely words in the output. (set to None if greedy decoding is enabled)
            repetition_penalty=1.2,  # Adjusts the likelihood of tokens that have already appeared in the text
            num_return_sequences=1,  # Number of alternative responses to generate
            pad_token_id=tokenizer.eos_token_id  # Padding token (end of sequence) to avoid generating unnecessary tokens after the prompt is finished
        )
    
    gen_time = time.time() - gen_start_time
    
    # Track memory after generation
    if torch.cuda.is_available() and device != 'cpu':
        memory_after_generation = torch.cuda.memory_allocated() / 1024**2
        print(f"GPU Memory after generation: {memory_after_generation:.2f} MB (Δ +{memory_after_generation - memory_before:.2f} MB)")
        
        # Add warning if approaching memory limits
        if memory_after_generation > 5000:  # Warning at ~5GB for 6GB card
            print("⚠️ WARNING: Memory usage is high. Consider resetting conversation with 'reset' command.")
    
    # Process output - extract only the model's response
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Add the assistant's response to the conversation history
    if remember_history:
        # Extract just the model's answer by removing the prompt
        assistant_response = generated_text
        # Some models repeat the instruction, try to remove it if present
        for prefix in ["Assistant:", "AI:"]:
            if assistant_response.startswith(prefix):
                assistant_response = assistant_response[len(prefix):].strip()
        
        # Add to history for next round
        conversation_history += f"\nAssistant: {assistant_response}"
    
    print(f"\nResponse: {generated_text}")
    print(f"Generation completed in {gen_time:.2f} seconds")

# Main execution
try:
    # Load the model with stable configuration
    print("Loading model...")
    load_start_time = time.time()
    
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        quantization_config=BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        ),
        device_map="auto",
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
    
    # Check if we should run in interactive mode or single prompt mode
    if args.interactive or args.prompt is None:
        print("\n===== Interactive Mode =====")
        print("Type 'exit' to quit the program\n")
        
        while True:
            user_input = input("Enter your prompt (or type 'exit' to quit, 'reset' to clear history): ")
            if user_input.lower() == 'exit':
                print("Exiting interactive mode...")
                break
            elif user_input.lower() == 'reset':
                conversation_history = ""
                print("Conversation history has been reset.")
                continue
            generate_response(user_input, model, tokenizer)
    else:
        # Single prompt mode
        generate_response(args.prompt, model, tokenizer)

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
    
    # Check if we should run in interactive mode or single prompt mode
    if args.interactive or args.prompt is None:
        print("\n===== Interactive Mode (CPU) =====")
        print("Type 'exit' to quit the program\n")
        
        while True:
            user_input = input("Enter your prompt (or type 'exit' to quit): ")
            if user_input.lower() == 'exit':
                print("Exiting interactive mode...")
                break
            generate_response(user_input, model, tokenizer, device='cpu')
    else:
        # Single prompt mode
        generate_response(args.prompt, model, tokenizer, device='cpu')