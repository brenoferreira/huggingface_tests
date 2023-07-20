import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./models/llama27b_hf")
model = AutoModelForCausalLM.from_pretrained("./models/llama27b_hf")

prompt = "What's the capital city of France?"
result_length = 100
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    inputs["input_ids"],
    max_length=result_length, 
    num_beams=2, 
    no_repeat_ngram_size=2,
    early_stopping=True
)[0]

text = tokenizer.decode(outputs, max_length=result_length)
print("\nText:\n" + 100 * '-')
print(text)