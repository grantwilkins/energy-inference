from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, Pipeline
import torch
from pyJoules.energy_meter import EnergyContext
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from pyJoules.device.rapl_device import RaplPackageDomain
from pyJoules.handler.csv_handler import CSVHandler
from pyJoules.handler.pandas_handler import PandasHandler
import argparse
import datetime
import pandas as pd
from pynvml.smi import nvidia_smi
import os
import psutil
import time
import numpy as np
from scipy import stats
import subprocess


def find_current_cpu_core():
    return psutil.Process().cpu_num()


def tokenizer_pipeline(
    model_name: str,
    ctx: EnergyContext,
) -> tuple[Pipeline, AutoTokenizer, tuple[int, int]]:
    tokenizer_cpu_core = find_current_cpu_core()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_cpu_core = find_current_cpu_core()
    ctx.record(tag="model load")
    pipe = pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return pipe, tokenizer, (tokenizer_cpu_core, model_cpu_core)


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
        min_new_tokens=int(num_tokens * 0.9),
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
    parser.add_argument("--hf_name", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--system_name", type=str, default="Swing")
    parser.add_argument("--batch_size", type=int, default=8)
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
    prompt = """What are some effective strategies for managing stress and maintaining good mental health during challenging times, such as a pandemic, a break-up, or a personal crisis?"""
    pandas_handle = PandasHandler()
    if out_dir == ".":
        start_time = datetime.datetime.now().strftime("%H-%M-%S")
    else:
        start_time = out_dir.split("/")[-1]

    if "AMD" in subprocess.check_output("lscpu").decode():
        domains = [NvidiaGPUDomain(i) for i in range(num_gpus)]
    elif "Intel" in subprocess.check_output("lscpu").decode():
        domains = [RaplPackageDomain(0), RaplPackageDomain(1)]
        domains.extend([NvidiaGPUDomain(i) for i in range(num_gpus)])

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
        file.write(f"    prompt-C: {prompt[:50].strip()}\n")

    with EnergyContext(
        handler=pandas_handle,
        domains=domains,
        start_tag="tokenizer",
    ) as ctx:
        pipe, tokenizer, (tokenizer_core, pipeline_core) = tokenizer_pipeline(
            hf_name, ctx
        )
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
    df["CPU Core"] = [tokenizer_core, pipeline_core]
    for idx_gpus in range(num_gpus):
        df[f"Total Memory {idx_gpus}"] = nvidia_smi.getInstance().DeviceQuery(
            "memory.total"
        )["gpu"][idx_gpus]["fb_memory_usage"]["total"]
        df[f"Used Memory {idx_gpus}"] = nvidia_smi.getInstance().DeviceQuery(
            "memory.used"
        )["gpu"][idx_gpus]["fb_memory_usage"]["used"]
    df.to_csv(
        f"{model_name}-{args.system_name}-{num_gpus}.csv",
        mode="a",
        header=False,
        index=False,
    )
    output_token_lengths = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    for num_tokens in output_token_lengths:
        runtimes = []
        for iteration in range(100):
            pandas_handle = PandasHandler()
            idx_log = (num_tokens, iteration)
            with EnergyContext(
                handler=pandas_handle,
                domains=domains,
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
            # print(llm_output)
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
            df["Prompt"] = prompt[:50].strip()
            df["Output Tokens"] = num_output_tokens
            df["Batch Size"] = batch_size
            df["System Name"] = system_name
            df["CPU Core"] = cpu_core
            for idx_gpus in range(num_gpus):
                df[f"Total Memory {idx_gpus}"] = nvidia_smi.getInstance().DeviceQuery(
                    "memory.total"
                )["gpu"][idx_gpus]["fb_memory_usage"]["total"]
                df[f"Used Memory {idx_gpus}"] = nvidia_smi.getInstance().DeviceQuery(
                    "memory.used"
                )["gpu"][idx_gpus]["fb_memory_usage"]["used"]
            torch.cuda.empty_cache()
            df.to_csv(
                f"{model_name}-{args.system_name}-{num_gpus}.csv",
                mode="a",
                header=False,
                index=False,
            )
            if iteration > 0:
                runtimes.append(inference_runtime)
            mean_runtime = np.mean(runtimes)
            std_err = stats.sem(runtimes)
            z_critical = stats.norm.ppf((1 + 0.95) / 2)
            ci_half_width = z_critical * std_err
            # Break if we have more than 5 samples and the confidence interval half-width is less than 0.5
            if iteration > 5 and ci_half_width < 0.5:
                break
