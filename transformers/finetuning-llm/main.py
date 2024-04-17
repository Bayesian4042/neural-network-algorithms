from model import Model
import torch
from datasets import load_dataset
import bitsandbytes as bnb
from transformers import BitsAndBytesConfig
import transformers
from transformers import AutoTokenizer #
from transformers import AutoModelForCausalLM
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)

from generate import Generate
from training import Training
from dataset import Dataset

# configs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Quantization config to use 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load dataset
DATASET_NAME = "ChrisHayduk/Llama-2-SQL-Dataset" # datasets for instruction
dataset = Dataset(DATASET_NAME)


# Load Model and tokenizer
MODEL_NAME = "NousResearch/Llama-2-7b-hf" # model to use
model = Model(model_name=MODEL_NAME, quantization_config=quantization_config, device_map=device).get_model()
tokenizer = model.get_tokenizer()

# test generate on base model
Generate(model = model, tokenizer=tokenizer, device=device).generate(
    '''Below is an instruction that describes a SQL generation task, paired with an input that provides further context about the available table schemas. Write SQL code that appropriately answers the request.
        ### Instruction:
        What was the smallest crowd of vfl park?

        ### Input:
        CREATE TABLE table_name_83 (crowd INTEGER, venue VARCHAR)

        ### Response: 
    ''')

# Training
Training(model=model, tokenizer=tokenizer, dataset=dataset).train()
