# Why Finetuning ?
1. domain specific knowledge
2. perform on specific task

# LoRA (Low rank adaptation)
1. state of the art technique to finr tune LLM
2. Type of finetuning:
    1. Full finetuning:
        a. take a pretrain model and continue traning which update all the paramters of the model.
        b. It is expensive. GPT3 -> 175B parameters -> 1TB

3. LoRA allows model to update some weights : W = W0 + B.A (reduces weights to train)

4. PEFT is SOTA library for finetuning models using LoRA

# Quantization
1. projection of 32 bits parametes to 16 bits (basically reducing the the precision)
2. for ex: 100 B parameters -> 32 bits -> 4 bytes x 100 x 10 ^ 9 => ~1TB
3. According to research: more parameters with less precision works better.
1B and 16 bit quantization => 4B and 4 bit quantization

# QLoRA
1. Quanitzed LoRA
2. bitandbytes library