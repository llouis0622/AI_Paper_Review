from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

prompt = """
Translate English to French.
English: cat
French: chat
English: dog
French: chien
English: house
French:
"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_new_tokens=10,
    do_sample=False
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
