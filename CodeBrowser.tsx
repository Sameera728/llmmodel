# OpenAPI REST API Specification

The open-source GPT-style LLM platform exposes high-performance FastAPI endpoints for integration with web wrappers, mobile clients, or microservices.

---

## 🟢 Base URL
```
http://localhost:8000
```

---

## 📡 1. Conversational Chat Completion

Similar to the OpenAI Chat API, this endpoint handles user-assistant exchanges with optional context and streaming.

* **Endpoint**: `POST /api/chat`
* **Content-Type**: `application/json`

### Request Payload
```json
{
  "messages": [
    {"role": "system", "content": "You are a specialized medical coding assistant."},
    {"role": "user", "content": "What is the ICD-10 code for acute appendicitis?"}
  ],
  "model": "qwen-2.5",
  "stream": false,
  "temperature": 0.2,
  "max_tokens": 512
}
```

### Response (Stream = False)
```json
{
  "model": "qwen-2.5",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "The ICD-10-CM code for acute appendicitis with generalized peritonitis is K35.20."
      }
    }
  ],
  "usage": {
    "prompt_tokens": 32,
    "completion_tokens": 24,
    "total_tokens": 56
  }
}
```

### Response (Stream = True)
Returns a standard Server-Sent Events (SSE) data stream.
```
data: The
data: ICD-10
data: code
data: for
data: acute
data: appendicitis
...
```

---

## 💾 2. Document Ingestion (RAG)

Accepts files to split, vectorize, and index into the FAISS memory layer.

* **Endpoint**: `POST /api/upload`
* **Content-Type**: `multipart/form-data`

### Multipart Fields
* `file`: Binary file upload (PDF, DOCX, TXT, or MD).

### Response
```json
{
  "success": true,
  "filename": "appendix_guidelines.pdf",
  "file_size_bytes": 1048576,
  "chunks_generated": 142,
  "embedded_index": "FAISS_Index_v1",
  "message": "Successfully indexed 'appendix_guidelines.pdf' for local context retrieval."
}
```

---

## ⚙️ 3. Model Orchestration

Query and dynamically hot-swap active baseline weights in the GPU cluster.

### 3.1 List Configured Models
* **Endpoint**: `GET /api/models`
* **Response**
```json
{
  "active_model": "qwen-2.5",
  "supported_models": ["llama-3", "mistral-7b", "qwen-2.5", "gemma-2", "phi-3"]
}
```

### 3.2 Hot-Swap Model
* **Endpoint**: `POST /api/models/switch`
* **Request Payload**
```json
{
  "model_name": "llama-3"
}
```
* **Response**
```json
{
  "success": true,
  "active_model": "llama-3",
  "message": "Successfully loaded inference context for llama-3"
}
```

---

## 📈 4. Fine-Tuning Job Management

### 4.1 Trigger Fine-Tuning Job
* **Endpoint**: `POST /api/train/start`
* **Request Payload**
```json
{
  "dataset_name": "medical_alpaca.json",
  "epochs": 3,
  "batch_size": 4,
  "learning_rate": 0.0002,
  "use_qlora": true
}
```
* **Response**
```json
{
  "success": true,
  "task_id": "job_ft_9872",
  "status": "queued",
  "message": "Fine-tuning job successfully initiated in background."
}
```

### 4.2 Query Job Status
* **Endpoint**: `GET /api/train/status/{task_id}`
* **Response**
```json
{
  "task_id": "job_ft_9872",
  "status": "running",
  "current_epoch": 1.4,
  "loss": 0.428,
  "steps": 140,
  "total_steps": 300,
  "eta_seconds": 1240
}
```
