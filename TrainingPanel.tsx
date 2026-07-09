# Open-Source GPT-Style LLM Configuration Manager
# Supports switching active models and custom fine-tuning/inference parameters

active_model: "qwen-2.5" # Options: llama-3, mistral-7b, qwen-2.5, gemma-2, phi-3

models:
  llama-3:
    name: "meta-llama/Meta-Llama-3-8B-Instruct"
    path: "./models/llama-3-8b"
    context_length: 8192
    device_map: "auto"
    torch_dtype: "float16" # or bfloat16
    load_in_8bit: false
    load_in_4bit: true

  mistral-7b:
    name: "mistralai/Mistral-7B-Instruct-v0.3"
    path: "./models/mistral-7b"
    context_length: 32768
    device_map: "auto"
    torch_dtype: "bfloat16"
    load_in_8bit: false
    load_in_4bit: true

  qwen-2.5:
    name: "Qwen/Qwen2.5-7B-Instruct"
    path: "./models/qwen-2.5-7b"
    context_length: 131072
    device_map: "auto"
    torch_dtype: "bfloat16"
    load_in_8bit: false
    load_in_4bit: true

  gemma-2:
    name: "google/gemma-2-9b-it"
    path: "./models/gemma-2-9b"
    context_length: 8192
    device_map: "auto"
    torch_dtype: "bfloat16"
    load_in_8bit: false
    load_in_4bit: true

  phi-3:
    name: "microsoft/Phi-3-mini-4k-instruct"
    path: "./models/phi-3-mini"
    context_length: 4096
    device_map: "auto"
    torch_dtype: "float16"
    load_in_8bit: false
    load_in_4bit: false

# Fine-Tuning Hyperparameters (LoRA & QLoRA)
training:
  output_dir: "./checkpoints"
  num_train_epochs: 3
  per_device_train_batch_size: 4
  per_device_eval_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  weight_decay: 0.01
  warmup_ratio: 0.03
  lr_scheduler_type: "cosine"
  logging_steps: 10
  save_strategy: "epoch"
  evaluation_strategy: "epoch"
  seed: 42
  fp16: false
  bf16: true
  report_to: "tensorboard"
  resume_from_checkpoint: true

# Parameter Efficient Fine-Tuning Settings (LoRA)
lora:
  r: 16
  lora_alpha: 32
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
  lora_dropout: 0.05
  bias: "none"
  task_type: "CAUSAL_LM"

# Retrieval Augmented Generation (RAG) Configuration
rag:
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  faiss_index_path: "./datasets/faiss_index"
  chunk_size: 500
  chunk_overlap: 50
  top_k: 4
  similarity_threshold: 0.7

# API Server Settings
api:
  host: "0.0.0.0"
  port: 8000
  debug: true
  workers: 2
  allowed_origins: ["*"]
