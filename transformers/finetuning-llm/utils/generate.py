import torch

class Generate:
  def __init__(self, model, tokenizer, device):
    self.model = model
    self.tokenizer = tokenizer
    self.generation_config = self.model.generation_config
    self.generation_config.pad_token_id = tokenizer.eos_token_id
    self.generation_config.eos_token_id = tokenizer.eos_token_id
    self.generation_config.max_new_tokens = 256 # GPT-2 has context length of 1024
    self.generation_config.temperature = 0.7
    self.generation_config.top_p = 0.9 # top p probability in the list
    self.generation_config.do_sample = True
    self.device = device

  def generate(self, prompt):
    self.generation_config.max_new_tokens = 256

    encoded = self.tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt").to(self.device)
    with torch.no_grad():
      out = self.model.generate(input_ids=encoded, generation_config=self.generation_config, repetition_penalty=2.0) # sampling algo, repitition giberish -> penality

    string_decoded = self.tokenizer.decode(out[0].tolist(), clean_up_tokenization_spaces=True)
    return string_decoded
  

def construct_datapoint(x, tokenizer):
  combined = x['input'] + x['output']
  return tokenizer(combined, padding=True)