# Upgrade pip
pip install --upgrade pip

# Install PyTorch (choose the correct command from https://pytorch.org/get-started/locally/)
pip install torch torchvision torchaudio

# Install Transformers (make sure it’s version 4.43.2 or later for LLaMA 3.1 support)
pip install transformers

# (Optional) Install bitsandbytes if you want to run the model in 8-bit mode
pip install bitsandbytes
