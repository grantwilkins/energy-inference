from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from pyJoules.energy_meter import EnergyContext
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from pyJoules.handler.csv_handler import CSVHandler

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--num_tokens", type=int, default=200)
parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-chat-hf")

args = parser.parse_args()


num_gpus = torch.cuda.device_count()
stats_name = args.model_name.split("/")[1]
csv_handle = CSVHandler(f"{stats_name}-{num_gpus}.csv")
num_tokens = args.num_tokens

with EnergyContext(
    handler=csv_handle,
    domains=[NvidiaGPUDomain(i) for i in range(num_gpus)],
    start_tag="tokenizer",
) as ctx:
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ctx.record(tag="pipeline load")

    pipe = pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    for i in range(10):
        ctx.record(tag=f"inference-{i}")
        prompt = "As a data scientist, can you explain the concept of regularization in machine learning?"
        sequences = pipe(
            prompt,
            do_sample=True,
            max_new_tokens=num_tokens,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
            batch_size=8192,
        )
        print(sequences[0]["generated_text"])

csv_handle.save_data()
