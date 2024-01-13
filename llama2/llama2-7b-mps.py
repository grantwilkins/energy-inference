from transformers import AutoTokenizer
import transformers
import torch

import time

model = "meta-llama/Llama-2-7b-chat-hf"

start_time = time.time()

tokenizer = AutoTokenizer.from_pretrained(model)

tokenizer_time = time.time() - start_time
print(f"Tokenizer loading time: {tokenizer_time} seconds")

start_time = time.time()

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

pipeline_time = time.time() - start_time
print(f"Pipeline loading time: {pipeline_time} seconds")

start_time = time.time()

sequences = pipeline(
    'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
    do_sample=True,
    top_k=10,
    num_return_sequences=2,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)

inference_time = time.time() - start_time
print(f"Inference time: {inference_time} seconds")

for seq in sequences:
    print(f"Result: {seq['generated_text']}")

total_tokens = sum(len(seq["generated_text"].split()) for seq in sequences)
token_rate = total_tokens / (tokenizer_time + pipeline_time + inference_time)

print(f"Total tokens: {total_tokens}")
print(f"Token rate: {token_rate} tokens per second")
