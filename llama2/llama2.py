import torch
from transformers import Llama2Model, Llama2Tokenizer

# Set the CUDA devices
device_ids = [0, 1, 2]  # Replace with the device IDs you want to use
torch.cuda.set_device(device_ids[0])
torch.cuda.init()

# Load the Llama 2 model and tokenizer
model_name = "llama-2"
model = Llama2Model.from_pretrained(model_name)
tokenizer = Llama2Tokenizer.from_pretrained(model_name)

# Move the model to CUDA devices
model = model.to(device_ids[0])
if len(device_ids) > 1:
    model = torch.nn.DataParallel(model, device_ids=device_ids)

# Accept prompts and process them
while True:
    prompt = input("Enter a prompt (or 'exit' to quit): ")
    if prompt == "exit":
        break

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to(device_ids[0])

    with torch.no_grad():
        outputs = model(**inputs)

    # Process the outputs as needed
    # ...
