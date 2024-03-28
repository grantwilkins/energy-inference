from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, Pipeline
import torch
from pyJoules.energy_meter import EnergyContext
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from pyJoules.handler.csv_handler import CSVHandler
from pyJoules.handler.pandas_handler import PandasHandler
import argparse
import datetime
import pandas as pd
from pynvml.smi import nvidia_smi
import psutil
import os
import time
import numpy as np
from scipy import stats


def find_current_cpu_core():
    return psutil.Process().cpu_num()


def tokenizer_model_pipeline(
    model_name: str,
    ctx: EnergyContext,
) -> tuple[Pipeline, AutoTokenizer, tuple[int, int, int]]:
    tokenizer_cpu_core = find_current_cpu_core()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_cpu_core = find_current_cpu_core()
    ctx.record(tag="model load")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    pipeline_cpu_core = find_current_cpu_core()
    ctx.record(tag="pipeline load")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return pipe, tokenizer, (tokenizer_cpu_core, model_cpu_core, pipeline_cpu_core)


def run_inference(
    pipe: Pipeline,
    num_tokens: int,
    prompt: str,
    batch_size: int,
) -> str:
    sequences = pipe(
        prompt,
        do_sample=True,
        max_new_tokens=num_tokens,
        min_new_tokens=num_tokens * 0.9,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
        batch_size=batch_size,
        use_cache=False,
    )
    return sequences[0]["generated_text"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_tokens", type=int, default=64)
    parser.add_argument("--hf_name", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--system_name", type=str, default="Swing")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out_dir", type=str, default=".")

    args = parser.parse_args()

    todays_date = datetime.date.today().strftime("%Y-%m-%d")
    num_gpus = torch.cuda.device_count()
    hf_name = args.hf_name
    model_name = hf_name.split("/")[1]
    num_tokens = args.num_tokens
    batch_size = args.batch_size
    system_name = args.system_name
    out_dir = args.out_dir
    prompts = {
        "A": "What is the capital of France?",
        "B": "Can you explain the difference between a simile and a metaphor? Provide an example of each.",
        "C": "What are some effective strategies for managing stress and maintaining good mental health during challenging times, such as a pandemic or a personal crisis?",
        "D": "Imagine you are a travel guide. Can you recommend a 7-day itinerary for a trip to Japan, including must-visit destinations, cultural experiences, and local cuisine? Provide a brief description of each day's activities and how they showcase the best of Japan.",
        "E": "As an AI language model, you have the ability to process and generate human-like text. Can you discuss the potential implications of advanced AI systems like yourself on various industries, such as healthcare, education, and creative fields? Consider the benefits, challenges, and ethical considerations surrounding the integration of AI in these sectors. Provide specific examples to support your analysis.",
    }
    pandas_handle = PandasHandler()
    # profile_tokenizer = ProfileAMDEnergy(
    #     tag="tokenizer-model-pipeline",
    #     date=todays_date,
    #     model=model_name,
    #     system_name=args.system_name,
    #     num_gpus=num_gpus,
    #     num_tokens=num_tokens,
    #     batch_size=batch_size,
    # )
    # profile_tokenizer_proc = profile_tokenizer.start_profiling()
    if out_dir == ".":
        start_time = datetime.datetime.now().strftime("%H-%M-%S")
    else:
        start_time = out_dir.split("/")[-1]

    with open(f"{out_dir}/job_info.yaml", "w") as file:
        file.write("job:\n")
        file.write(f"  date: {todays_date}\n")
        file.write(f"  start_time: {start_time}\n")
        file.write("  details:\n")
        file.write(f"    model_name: {model_name}\n")
        file.write(f"    system_name: {system_name}\n")
        file.write(f"    num_tokens: {num_tokens}\n")
        file.write(f"    batch_size: {batch_size}\n")
        file.write(f"    hf_name: {hf_name}\n")
        for idx, prompt in prompts.items():
            file.write(f"    prompt-{idx}: {prompt}\n")

    with EnergyContext(
        handler=pandas_handle,
        domains=[NvidiaGPUDomain(i) for i in range(num_gpus)],
        start_tag="tokenizer",
    ) as ctx:
        pipe, tokenizer, cores = tokenizer_model_pipeline(args.hf_name, ctx)
    # profile_tokenizer.stop_profiling(proc=profile_tokenizer_proc)
    df = pandas_handle.get_dataframe()
    df["Max Number of Tokens"] = num_tokens
    df["Input Tokens"] = 0
    df["Iteration"] = 0
    df["Model Name"] = model_name
    df["Number of GPUs"] = num_gpus
    df["Prompt"] = "startup"
    df["Output Tokens"] = 0
    df["Batch Size"] = batch_size
    df["System Name"] = system_name
    for idx in range(num_gpus):
        df[f"Total Memory {idx}"] = nvidia_smi.getInstance().DeviceQuery(
            "memory.total"
        )["gpu"][idx]["fb_memory_usage"]["total"]
        df[f"Used Memory {idx}"] = nvidia_smi.getInstance().DeviceQuery("memory.used")[
            "gpu"
        ][idx]["fb_memory_usage"]["used"]
    df["CPU Core"] = cores
    df.to_csv(
        f"{model_name}-{args.system_name}-{num_gpus}.csv",
        mode="a",
        header=False,
        index=False,
    )

    for idx, prompt in prompts.items():
        runtimes = []
        for iteration in range(100):
            pandas_handle = PandasHandler()
            idx_log = (idx, iteration)
            # profile_inference = ProfileAMDEnergy(
            #     tag=f"inference-{idx_log[0]}-{idx_log[1]}",
            #     date=todays_date,
            #     model=model_name,
            #     system_name=args.system_name,
            #     num_gpus=num_gpus,
            #     num_tokens=num_tokens,
            #     batch_size=batch_size,
            # )
            # profile_inference_proc = profile_inference.start_profiling()
            with EnergyContext(
                handler=pandas_handle,
                domains=[NvidiaGPUDomain(i) for i in range(num_gpus)],
                start_tag=f"start-inference-{idx_log[0]}-{idx_log[1]}",
            ) as ctx:
                cpu_core = find_current_cpu_core()
                inference_start = time.time()
                llm_output = run_inference(
                    pipe=pipe,
                    num_tokens=num_tokens,
                    prompt=prompt,
                    batch_size=batch_size,
                )
                inference_end = time.time()
                inference_runtime = inference_end - inference_start
            print(llm_output)

            # profile_inference.stop_profiling(proc=profile_inference_proc)
            input_tokens = tokenizer.encode(prompt)
            num_input_tokens = len(input_tokens)
            output_tokens = tokenizer.encode(llm_output)
            num_output_tokens = len(output_tokens)
            df = pandas_handle.get_dataframe()
            df["Max Number of Tokens"] = num_tokens
            df["Input Tokens"] = num_input_tokens
            df["Iteration"] = iteration
            df["Model Name"] = model_name
            df["Number of GPUs"] = num_gpus
            df["Prompt"] = prompt[:50]
            df["Output Tokens"] = num_output_tokens
            df["Batch Size"] = batch_size
            df["System Name"] = system_name
            df["CPU Core"] = cpu_core
            for idx in range(num_gpus):
                df[f"Total Memory {idx}"] = nvidia_smi.getInstance().DeviceQuery(
                    "memory.total"
                )["gpu"][idx]["fb_memory_usage"]["total"]
                df[f"Used Memory {idx}"] = nvidia_smi.getInstance().DeviceQuery(
                    "memory.used"
                )["gpu"][idx]["fb_memory_usage"]["used"]

            df.to_csv(
                f"{model_name}-{args.system_name}-{num_gpus}.csv",
                mode="a",
                header=False,
                index=False,
            )
            runtimes.append(inference_runtime)
            mean_runtime = np.mean(runtimes)
            std_err = stats.sem(runtimes)
            z_critical = stats.norm.ppf((1 + 0.95) / 2)
            ci_half_width = z_critical * std_err
            # Break if we have more than 5 samples and the confidence interval half-width is less than 0.5
            if iteration > 5 and ci_half_width < 0.5:
                break
