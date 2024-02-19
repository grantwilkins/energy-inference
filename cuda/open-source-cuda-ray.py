from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from pyJoules.energy_meter import EnergyContext
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from pyJoules.handler.csv_handler import CSVHandler
import argparse
import ray

parser = argparse.ArgumentParser()

parser.add_argument("--num_tokens", type=int, default=200)
parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-v0.1")

args = parser.parse_args()

ray.init(address="auto")

num_gpus = torch.cuda.device_count()
stats_name = args.model_name.split("/")[1]
csv_handle = CSVHandler(f"{stats_name}-{num_gpus}.csv")
num_tokens = args.num_tokens
# domains = NvididaGPUDomain0 if num_gpus == 1 else [i for i in range(num_gpus)]


@ray.remote
def run_inference(model_name, prompt, num_tokens, ctx):

    # Load model and tokenizer
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ctx.record(tag="model load")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    ctx.record(tag="pipeline load")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Generate text
    ctx.record(tag=f"inference-1")
    sequences = pipe(
        prompt,
        do_sample=True,
        max_new_tokens=num_tokens,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
    )

    return sequences[0]["generated_text"]


with EnergyContext(
    handler=csv_handle,
    domains=[NvidiaGPUDomain(i) for i in range(num_gpus)],
    start_tag="tokenizer",
) as ctx:
    prompt = "As a data scientist, can you explain the concept of regularization in machine learning?"
    results = ray.get(
        run_inference.remote("mistralai/Mistral-7B-v0.1", prompt, num_tokens, ctx)
    )
    print(results)

csv_handle.save_data()
