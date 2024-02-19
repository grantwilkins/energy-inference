from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from pyJoules.energy_meter import EnergyContext
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from pyJoules.handler.csv_handler import CSVHandler
from llm_serving.model.wrapper import get_model
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument("--num_tokens", type=int, default=200)
parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-v0.1")

args = parser.parse_args()


num_gpus = torch.cuda.device_count()
stats_name = args.model_name.split("/")[1]
csv_handle = CSVHandler(f"{stats_name}-{num_gpus}.csv")
num_tokens = args.num_tokens
# domains = NvididaGPUDomain0 if num_gpus == 1 else [i for i in range(num_gpus)]

with EnergyContext(
    handler=csv_handle,
    domains=[NvidiaGPUDomain(i) for i in range(num_gpus)],
    start_tag="tokenizer",
) as ctx:
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ctx.record(tag="model load")
    model = get_model(
        model_name=model_name,
        path=os.environ["TRANSFORMERS_CACHE"],
    )
    # ctx.record(tag="pipeline load")

    for i in range(10):
        ctx.record(tag=f"inference-{i}")
        prompt = "As a data scientist, can you explain the concept of regularization in machine learning?"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        output = model.generate(input_ids=input_ids, max_length=256, do_sample=True)
        generated_string = tokenizer.batch_decode(output, skip_special_tokens=True)

csv_handle.save_data()
