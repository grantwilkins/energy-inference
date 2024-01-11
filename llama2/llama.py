import torch
from transformers import LlamaModel, LlamaTokenizer

mps_device = torch.device("mps")
# Load the Llama 2 model and tokenizer
model_name = "llama"
model = LlamaModel.from_pretrained(model_name)
tokenizer = LlamaTokenizer.from_pretrained(model_name)

# Move the model to CUDA devices
model = model.to(mps_device)

# Accept prompts and process them
while True:
    prompt = input("Enter a prompt (or 'exit' to quit): ")
    if prompt == "exit":
        break

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to(mps_device)

    with torch.no_grad():
        outputs = model(**inputs)
