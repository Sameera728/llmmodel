# Modular LLM Fine-Tuning Script
# Utilizes PyTorch, Transformers, Accelerate, PEFT (LoRA/QLoRA), and BitsAndBytes.

import os
import sys
import yaml
import torch
import logging
from typing import Dict, Any

# HuggingFace core packages (assumed in cluster env)
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     BitsAndBytesConfig,
#     TrainingArguments,
#     Trainer,
#     DataCollatorForSeq2Seq
# )
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Logger Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("train")

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_training(config_path: str = "./configs/config.yaml"):
    logger.info(f"Loading configuration from: {config_path}")
    config = load_yaml_config(config_path)
    
    active_model_name = config["active_model"]
    model_cfg = config["models"][active_model_name]
    train_cfg = config["training"]
    lora_cfg = config["lora"]
    
    logger.info(f"Preparing pipeline for Model: {model_cfg['name']}")
    
    # 1. Quantization Configuration (QLoRA setup)
    # Under standard PyTorch workflows, BitsAndBytes defines 4-bit NormalFloat:
    bnb_config = None
    if model_cfg.get("load_in_4bit"):
        logger.info("Initializing 4-bit NormalFloat Quantization schema (NF4)")
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16 if train_cfg.get("bf16") else torch.float16
        # )
    
    # 2. Tokenizer Initialization
    logger.info(f"Downloading/loading Tokenizer for: {model_cfg['name']}")
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_cfg["name"],
    #     trust_remote_code=True
    # )
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "right" # Recommended side for AutoRegressive Causal LMs

    # 3. Model Loading with device_map placement
    logger.info("Loading pre-trained base model weights...")
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_cfg["name"],
    #     quantization_config=bnb_config,
    #     device_map=model_cfg["device_map"],
    #     torch_dtype=torch.bfloat16 if train_cfg.get("bf16") else torch.float16,
    #     trust_remote_code=True
    # )
    
    # 4. Prepare Model for Gradient Checkpointing and K-Bit Training
    # if model_cfg.get("load_in_4bit"):
    #     model = prepare_model_for_kbit_training(model)
        
    # 5. Initialize LoRA Adapters
    logger.info("Configuring LoRA Adapter parameters...")
    # peft_config = LoraConfig(
    #     r=lora_cfg["r"],
    #     lora_alpha=lora_cfg["lora_alpha"],
    #     target_modules=lora_cfg["target_modules"],
    #     lora_dropout=lora_cfg["lora_dropout"],
    #     bias=lora_cfg["bias"],
    #     task_type=lora_cfg["task_type"]
    # )
    # model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters()
    
    # 6. Load Datasets & Apply Tokenization (Placeholder structure representing dataset format mapping)
    logger.info("Preprocessing training and validation data streams...")
    # From dataset script:
    # train_dataset = load_and_tokenize_dataset(tokenizer, config)
    # eval_dataset = load_and_tokenize_dataset(tokenizer, config, is_eval=True)
    
    # 7. Configure TrainingArguments
    logger.info("Setting up HuggingFace Trainer arguments...")
    # training_args = TrainingArguments(
    #     output_dir=train_cfg["output_dir"],
    #     num_train_epochs=train_cfg["num_train_epochs"],
    #     per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
    #     per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
    #     gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
    #     learning_rate=float(train_cfg["learning_rate"]),
    #     weight_decay=train_cfg["weight_decay"],
    #     warmup_ratio=train_cfg["warmup_ratio"],
    #     lr_scheduler_type=train_cfg["lr_scheduler_type"],
    #     logging_steps=train_cfg["logging_steps"],
    #     evaluation_strategy=train_cfg["evaluation_strategy"],
    #     save_strategy=train_cfg["save_strategy"],
    #     fp16=train_cfg["fp16"],
    #     bf16=train_cfg["bf16"],
    #     report_to=train_cfg["report_to"],
    #     seed=train_cfg["seed"],
    #     logging_dir=os.path.join(train_cfg["output_dir"], "runs")
    # )
    
    # 8. Initialize Trainer and Begin Optimization loop
    logger.info("Initializing HuggingFace Trainer engine...")
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    # )
    
    # logger.info("Starting Fine-tuning loop execution...")
    # trainer.train(resume_from_checkpoint=train_cfg.get("resume_from_checkpoint", True))
    
    # 9. Save final adapters
    final_adapter_path = os.path.join(train_cfg["output_dir"], "final_adapters")
    # model.save_pretrained(final_adapter_path)
    # tokenizer.save_pretrained(final_adapter_path)
    
    logger.info(f"Training completed successfully! Saved LoRA adapters to: {final_adapter_path}")

if __name__ == "__main__":
    config_file = "./configs/config.yaml"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    run_training(config_file)
