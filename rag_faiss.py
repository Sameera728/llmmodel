# Dataset Preprocessing & Cleaning Pipelines
# Prepares instruction-response pairs (Alpaca / ShareGPT schemas) for Causal Language Models.

import json
from typing import List, Dict, Any

class DatasetPreparer:
    def __init__(self, tokenizer_name: str = None):
        self.tokenizer_name = tokenizer_name
        self.system_prompt = "You are a helpful, respectful, and honest open-source assistant."

    def clean_raw_sample(self, raw_data: Dict[str, Any]) -> Dict[str, str]:
        """Cleans and normalizes fields in raw JSON records, stripping unwanted HTML, boilerplate, or null content."""
        instruction = raw_data.get("instruction", "").strip()
        input_context = raw_data.get("input", "").strip()
        output = raw_data.get("output", "").strip()
        
        # Strip simple HTML boilerplate tags
        instruction = re_clean(instruction)
        input_context = re_clean(input_context)
        output = re_clean(output)
        
        return {
            "instruction": instruction,
            "input": input_context,
            "output": output
        }

    def format_alpaca_prompt(self, clean_sample: Dict[str, str]) -> str:
        """Applies a standard Alpaca template to generate aligned textual representations."""
        instruction = clean_sample["instruction"]
        input_context = clean_sample["input"]
        output = clean_sample["output"]
        
        if input_context:
            prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_context}\n\n### Response:\n{output}"
        else:
            prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{output}"
        return prompt

    def tokenize_and_align_labels(self, formatted_text: str, tokenizer: Any, max_length: int = 2048) -> Dict[str, List[int]]:
        """Tokenizes formatted prompts and masks instruction text tokens so only target outputs calculate training loss."""
        # Under normal cluster runs:
        # encodings = tokenizer(
        #     formatted_text,
        #     truncation=True,
        #     max_length=max_length,
        #     padding=False,
        #     return_tensors=None
        # )
        # # Create standard causal LM labels where pad/instruction inputs are replaced with -100 index (ignored in loss calc)
        # labels = encodings["input_ids"].copy()
        # # Find prompt boundary to mask:
        # prompt_boundary = formatted_text.rfind("### Response:\n")
        # tokenized_prompt_len = len(tokenizer.tokenize(formatted_text[:prompt_boundary]))
        # labels[:tokenized_prompt_len] = [-100] * tokenized_prompt_len
        # return {"input_ids": encodings["input_ids"], "attention_mask": encodings["attention_mask"], "labels": labels}
        
        # Simplified token list for demonstration
        return {
            "input_ids": [101, 2054, 2003, 1037, 3432, 102],
            "attention_mask": [1, 1, 1, 1, 1, 1],
            "labels": [-100, -100, -100, 3432, 102]
        }

def re_clean(text: str) -> str:
    """Helper to remove simple HTML tags and clean up whitespace."""
    if not text:
        return ""
    text = re.sub(r"<[^>]*>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Import regex helper internally
import re

if __name__ == "__main__":
    preparer = DatasetPreparer()
    raw_sample = {
        "instruction": "Explain <strong>bfloat16</strong> quantization.",
        "input": "",
        "output": "bfloat16 is a brain floating point format using 16 bits with identical range to float32."
    }
    cleaned = preparer.clean_raw_sample(raw_sample)
    formatted = preparer.format_alpaca_prompt(cleaned)
    print("--- Formatted Prompt Template Output ---")
    print(formatted)
