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

