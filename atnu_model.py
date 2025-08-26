import os
import json
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from tqdm import tqdm
import re
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    DataCollatorForLanguageModeling,
    Trainer
)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

## Data Loading and Preparation

def load_json_data(file_path):
    """Load JSON data from file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# Load training, dev, and test data
train_data = load_json_data('train.json')
dev_data = load_json_data('dev.json')
test_data = load_json_data('test.json')

# Sample inspection
print(f"Number of training examples: {len(train_data)}")
print(f"Number of dev examples: {len(dev_data)}")
print(f"Number of test examples: {len(test_data)}")
print("\nSample training example:")
print(json.dumps(train_data[0], indent=2))

## Data Preprocessing

def format_example_for_sft(example):
    """Format a single example for supervised fine-tuning."""
    tokens = example['tokens']
    sentence = ' '.join(tokens)
    
    # For training and dev data (which have entities)
    if 'entities' in example:
        entities = example['entities']
        entity_spans = [entity['span'] for entity in entities]
        entity_str = ', '.join(entity_spans)
        
        # Instruction format for NER task
        instruction = f"Extract all disease entities from the following sentence."
        input_text = f"{sentence}"
        output_text = f"{entity_str}"
        
        return {
            "id": example['id'],
            "instruction": instruction,
            "input": input_text,
            "output": output_text,
            "sentence": sentence,
            "entities": entity_spans
        }
    else:
        # For test data (no entities provided)
        instruction = f"Extract all disease entities from the following sentence."
        input_text = f"{sentence}"
        
        return {
            "id": example['id'],
            "instruction": instruction,
            "input": input_text,
            "output": "",  # No output for test data
            "sentence": sentence,
            "entities": []
        }

# Process the datasets
train_processed = [format_example_for_sft(example) for example in train_data]
dev_processed = [format_example_for_sft(example) for example in dev_data]
test_processed = [format_example_for_sft(example) for example in test_data]

# Convert to dataset format
train_dataset = Dataset.from_pandas(pd.DataFrame(train_processed))
dev_dataset = Dataset.from_pandas(pd.DataFrame(dev_processed))
test_dataset = Dataset.from_pandas(pd.DataFrame(test_processed))

print(f"Processed {len(train_dataset)} training examples")
print(f"Processed {len(dev_dataset)} dev examples")
print(f"Processed {len(test_dataset)} test examples")
