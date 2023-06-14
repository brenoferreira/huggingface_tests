import torch
from transformers import BloomForCausalLM, BloomTokenizerFast

tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")
model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m")

prompt = "It was a dark and stormy night"
result_length = 50
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    inputs["input_ids"],
    max_length=result_length, 
    num_beams=2, 
    no_repeat_ngram_size=2,
    early_stopping=True
)[0]


print("Output:\n" + 100 * '-')
print(outputs)

text = tokenizer.decode(outputs, max_length=result_length)
print("\nText:\n" + 100 * '-')
print(text)