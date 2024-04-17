import torch
from datasets import load_dataset
import bitsandbytes as bnb
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer #
from transformers import AutoModelForCausalLM
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)


class Model:
    def __init__(self, model_name, quantization_config, device_map="auto"):
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.device_map = device_map
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device_map,
        )
        self.model.use_cache = True
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    def prepare_model_for_kbit_training(self, peft_config):
        model = prepare_model_for_kbit_training(self.model)
        model = get_peft_model(model, peft_config)
        return model

    def construct_datapoint(self, example):
        return construct_datapoint(example, self.tokenizer)

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def get_device_map(self):
        return self.device_map

    def get_model_name(self):
        return self.model_name

    def get_quantization_config(self):
        return self.quantization_config

    def get_model_config(self):
        return self.model.config

    def get_model_type(self):
        return self.model.config.model_type

    def get_model_class(self):
        return self.model.__class__

    def get_model_device(self):
        return self.model.device

    def get_model_dtype(self):
        return self.model.dtype

    def get_model_state_dict(self):
        return self.model.state_dict()

    def get_model_parameters(self):
        return self.model.parameters()

    def get_model_modules(self):
        return self.model.modules()

    def get_model_named_parameters(self):
        return self.model.named_parameters()

    def get_model_named_modules(self):
        return self.model.named_modules()

    def get_model_named_buffers(self):
        return self.model.named_buffers()

    def get_model_named_children(self):
        return self.model.named_children()

    def get_model_named_modules(self):
        return self.model.named_modules()

    def get_model_named_parameters(self):
        return self.model.named_parameters()

    def get_model_named_buffers(self):
        return self.model.named_buffers()

    def get_model_named_children(self):
        return self.model.named_children()

    def get_model_named_modules(self):
        return self.model.named_modules()



# Load model
MODEL_NAME = "NousResearch/Llama-2-7b-hf"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map="auto",
)

model.use_cache = True

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

concat_training_dataset = training_dataset.map(construct_datapoint)

# Prepare model for kbit training
peft_config = LoraConfig(
    r=16,
    lora_alpha=32, # 
    target_modules=["q_proj", "v_proj", "k_proj", "down_proj", "gate_proj", "o_proj", "up_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)