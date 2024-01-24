from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from pyJoules.energy_meter import EnergyContext
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from pyJoules.handler.csv_handler import CSVHandler

num_gpus = torch.cuda.device_count()
csv_handle = CSVHandler(f"llama2-70b-cuda-{num_gpus}.csv")
num_tokens = 200

with EnergyContext(
    handler=csv_handle,
    domains=[NvidiaGPUDomain(i) for i in range(num_gpus)],
    start_tag="tokenizer",
) as ctx:
    model_name = "meta-llama/Llama-2-70b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/lcrc/project/ECP-EZ/ac.gwilkins/")
    ctx.record(tag="pipeline load")

    pipe = pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    ctx.record(tag="inference")
    prompt = "As a data scientist, can you explain the concept of regularization in machine learning?"
    sequences = pipe(
        prompt,
        do_sample=True,
        max_new_tokens=num_tokens,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
    )
    print(sequences[0]["generated_text"])

csv_handle.save_data()
