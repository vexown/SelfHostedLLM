# SelfHostedLLM

License info:
    Meta Llama 3 8B Instruct
    Meta Llama 3 is licensed under the Meta Llama 3 Community License, 
    Copyright © Meta Platforms, Inc. All Rights Reserved.

    This script downloads and runs the Meta Llama 3 8B Instruct model.
    Please ensure you have read and agree to the license and Acceptable Use Policy.
    For more information, visit https://llama.meta.com/llama3

    This project downloads and runs the Meta Llama 3 8B Instruct model.
    Please ensure you have read and agree to the license and Acceptable Use Policy.
    For more information, visit https://llama.meta.com/llama3

Prerequisites:

1. Register on HuggingFace and request access to Llama 3 - https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

2. Create huggingface_token.txt in this directory with your token


Quick Start step-by-step:

1. ./prepare_env.sh

2. source llama_env/bin/activate

3. ./install_libraries.sh

4. python3 download_and_run_model.py