import torch
from transformers import BloomForCausalLM, BloomTokenizerFast

tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")
model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m")

LENGTH = 1000

def resposta(prompt, length):
  inputs = tokenizer(prompt, return_tensors="pt")
  outputs = model.generate(
      inputs["input_ids"],
      max_length=length, 
      num_beams=2, 
      no_repeat_ngram_size=2,
      early_stopping=True
  )[0]

  text = tokenizer.decode(outputs, max_length=length)
  return text


while True:
    prompt = input("Prompt (ou 'fim' para sair): ")
    if prompt.lower() == "fim":
        break
    else:
      res = resposta(prompt, LENGTH)

      print("Resposta:\n" + 100 * '-')

      print(res)

      print(100 * '-')