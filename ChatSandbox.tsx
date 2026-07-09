# Open-Source GPT-Style LLM Installation Guide

Welcome to the open-source LLM orchestrator. This guide will walk you through setting up the complete technology stack on Linux, Windows, or Docker.

---

## 🛠️ Hardware Requirements

For base inference or small-scale fine-tuning (LoRA/QLoRA) of models like LLaMA-3 (8B) or Mistral (7B):
* **CUDA GPU**: NVIDIA GPU with at least 16GB VRAM (e.g., RTX 3090, RTX 4090, A10G, or A100).
* **RAM**: 32GB System memory minimum.
* **Storage**: 100GB SSD/NVMe (for storing downloaded weights and model checkpoints).

---

## 🐧 1. Linux Setup (Ubuntu 22.04 LTS / 24.04 LTS)

Ensure you have installed the **NVIDIA Container Toolkit** and the latest **NVIDIA Drivers** (CUDA 12.1+ compatible).

### Step 1.1: System Dependencies
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.12 python3.12-venv python3-pip git build-essential curl
```

### Step 1.2: Virtual Environment Configuration
```bash
# Clone the repository
git clone https://github.com/yourusername/open-source-llm.git
cd open-source-llm

# Initialize virtualenv
python3.12 -m venv venv
source venv/bin/activate
```

### Step 1.3: Install Torch with CUDA 12.1 support
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 1.4: Install Package Requirements
```bash
pip install -r docker/requirements.txt
```

---

## 🪟 2. Windows 11 Setup (via WSL2)

It is highly recommended to run this project in **WSL2 (Windows Subsystem for Linux)** to support GPU acceleration with CUDA and BitsAndBytes bindings.

1. Ensure Windows Terminal and the **NVIDIA Windows Display Driver** are installed (they automatically bridge CUDA inside WSL2).
2. Install Ubuntu 22.04 inside WSL:
   ```cmd
   wsl --install -d Ubuntu-22.04
   ```
3. Open WSL terminal and follow the **Linux Setup** instructions above.

---

## 🐳 3. Docker Deployment (Recommended for Cloud/Production)

Docker Compose handles building the FastAPI server, bootstrapping PostgreSQL, and running TensorBoard inside isolated, CUDA-aware runtimes.

### Step 3.1: Install NVIDIA Container Toolkit
Ensure Docker has GPU access inside the daemon by following the [NVIDIA Setup Guidelines](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

### Step 3.2: Launch multi-container stack
```bash
cd docker
docker compose up --build -d
```

Confirm that all containers are healthy:
```bash
docker compose ps
```

---

## 🎯 4. Verification Check
Test that PyTorch successfully detects your GPU from inside your virtual environment:
```python
import torch
print("CUDA Available:", torch.cuda.is_available())
print("Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
```
If this prints `CUDA Available: True`, your environment is fully ready for local inference and fine-tuning!
