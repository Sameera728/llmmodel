# LoRA & QLoRA Fine-Tuning Guide

Fine-tuning allows you to specialize an open-source base model (such as Mistral-7B or Qwen-2.5) on domain-specific datasets (e.g., medical logs, legal guidelines, or custom customer service scripts). This workspace supports **LoRA (Low-Rank Adaptation)** and **QLoRA (Quantized Low-Rank Adaptation)**.

---

## 📊 Dataset Format

Datasets must be formatted as JSON arrays of objects following either the **Alpaca style** or the **conversational (ShareGPT) style**.

### 1. Alpaca Format (`dataset.json`)
```json
[
  {
    "instruction": "Convert the following temperature from Fahrenheit to Celsius.",
    "input": "Temperature: 100 F",
    "output": "To convert Fahrenheit to Celsius, use the formula C = (F - 32) * 5/9. Placed into the formula: C = (100 - 32) * 5/9 = 68 * 5/9 = 37.78 C."
  },
  {
    "instruction": "Define QLoRA in deep learning.",
    "input": "",
    "output": "QLoRA (Quantized Low-Rank Adaptation) is an efficient fine-tuning technique that quantizes a pre-trained base LLM down to 4-bit NormalFloat (NF4) while adding small, trainable 16-bit LoRA adapters."
  }
]
```

---

## ⚙️ Fine-Tuning Parameters

All training hyperparameters are configured in `configs/config.yaml`. The key parameters include:

| Parameter | Recommended Value | Explanation |
| :--- | :--- | :--- |
| `lora.r` | `16` or `32` | Rank size of the adapter matrices. Higher values capture more complex representations but increase VRAM. |
| `lora.lora_alpha` | `32` or `64` | Scaling factor for the adapter updates. Typically set to `2 * r`. |
| `lora.target_modules` | `[q_proj, v_proj, k_proj, o_proj]` | Target attention projections to inject adapter weights. |
| `training.learning_rate` | `2.0e-4` | Peak learning rate. QLoRA works best between `1e-4` and `3e-4`. |
| `training.per_device_train_batch_size` | `4` | Batch size per GPU. Adjust depending on available VRAM. |
| `training.gradient_accumulation_steps` | `4` | Number of backward steps to accumulate gradients before performing optimizer updates. |

---

## 🚀 Running the Fine-Tuning Pipeline

### Step 1: Pre-install Accelerate Config
Set up Hugging Face's multi-GPU and distributed clustering configuration:
```bash
accelerate config
```

### Step 2: Start Training
To start fine-tuning with the default configuration specified in `configs/config.yaml`:
```bash
python training/train.py
```
To run on multiple GPUs using Accelerate:
```bash
accelerate launch training/train.py configs/config.yaml
```

---

## 📈 TensorBoard Visualization

Monitor loss, learning rate decay, and evaluation performance in real-time.

1. Ensure the training is running and writing logs into `./checkpoints/runs`.
2. Start the TensorBoard server:
   ```bash
   tensorboard --logdir=./checkpoints/runs --port=6006
   ```
3. Open your browser and navigate to `http://localhost:6006`.

---

## 🔄 Checkpoints and Resuming Training

If your training run is interrupted by hardware timeout or node preemption, the script is configured to automatically search `./checkpoints` for the latest subfolder (e.g. `checkpoint-500/`) and resume seamlessly from where it stopped:
```yaml
training:
  resume_from_checkpoint: true
```
This saves state, model adapters, and optimizer momenta safely.
