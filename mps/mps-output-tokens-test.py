from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, Pipeline
import torch
import subprocess
import threading
import re
import pandas as pd
from time import sleep
import time
import argparse
import os
import datetime
import torch.mps
import numpy as np
from scipy import stats
import random


def load_model(
    model_name: str,
    load_model_event: threading.Event,
    load_model_thread: threading.Thread,
) -> tuple[Pipeline, AutoTokenizer, float]:

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    load_model_thread.start()
    load_model_start_time = time.time()
    pipe = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    load_model_time = time.time() - load_model_start_time
    load_model_event.set()
    load_model_thread.join()

    return (
        pipe,
        tokenizer,
        load_model_time,
    )


def run_inference(
    pipe: Pipeline,
    num_tokens: int,
    prompt: str,
    batch_size: int,
    inference_event: threading.Event,
    inference_monitor: threading.Thread,
) -> str:
    inference_monitor.start()
    sleep(2)
    sequences = pipe(
        prompt,
        do_sample=True,
        max_new_tokens=num_tokens,
        min_new_tokens=int(num_tokens * 0.9),
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
        use_cache=False,
        batch_size=batch_size,
    )
    inference_event.set()
    inference_monitor.join()

    return sequences[0]["generated_text"]


def monitor_power_usage(power_readings: list[str], stop_monitoring: threading.Event):
    cmd = "echo ***REMOVED*** | sudo -S powermetrics --show-process-energy -i 200"
    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, shell=True, text=True
    ) as process:
        while not stop_monitoring.is_set():
            line = process.stdout.readline()
            if not line:
                break
            power_readings.append(line)
    print("Powermetrics process ended.")


def process_data(power_readings, event):
    cpu_pattern = r"CPU Power:\s+(\d+)\s+mW"
    gpu_pattern = r"GPU Power:\s+(\d+)\s+mW"
    keywords = ["python3.12", "ALL_TASKS"]
    readings = []
    power_usage = {"event": event}
    for line in power_readings:
        if not line:
            break
        cpu_match = re.search(cpu_pattern, line)
        gpu_match = re.search(gpu_pattern, line)
        if any(line.startswith(keyword) for keyword in keywords):
            split_line = line.split()
            power_usage[split_line[0]] = split_line[-1]
        if line.startswith("*** Sampled"):
            power_usage["timestamp"] = float(
                line.split()[-3].replace("(", "").replace("ms", "")
            )
        if cpu_match:
            power_usage["cpu"] = int(cpu_match.group(1))
        if gpu_match:
            power_usage["gpu"] = int(gpu_match.group(1))
        if power_usage.keys() == {
            "cpu",
            "gpu",
            "python3.12",
            "ALL_TASKS",
            "event",
            "timestamp",
        }:
            readings.append(power_usage)
            power_usage = {"event": event}
    return readings


def post_process_power_data(readings: list[dict], runtime_s: float) -> pd.DataFrame:
    df = pd.DataFrame()
    df["Event"] = [readings[0]["event"]]
    df["Runtime (s)"] = [runtime_s]
    total_cpu_energy = 0
    for reading in readings:
        total_cpu_energy += (
            reading["cpu"]
            * reading["timestamp"]
            * (float(reading["python3.12"]) / float(reading["ALL_TASKS"]))
            / 1000
            / 1000
        )
    df["CPU Energy (J)"] = [total_cpu_energy]
    total_gpu_energy = 0
    for reading in readings:
        total_gpu_energy += reading["gpu"] * reading["timestamp"] / 1000 / 1000
    df["GPU Energy (J)"] = [total_gpu_energy]
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_tokens", type=int, default=32)
    parser.add_argument("--hf_name", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--system_name", type=str, default="M1-Pro")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    todays_date = datetime.date.today().strftime("%Y-%m-%d")
    start_time = datetime.datetime.now().strftime("%H-%M-%S")
    num_tokens = args.num_tokens
    hf_name = args.hf_name
    system_name = args.system_name
    batch_size = args.batch_size
    model_name = args.hf_name.split("/")[1]
    prompt = """What are some effective strategies for managing stress and maintaining good mental health during challenging times, such as a pandemic, a break-up, or a personal crisis?"""
    out_dir = f"{model_name}/{todays_date}/{start_time}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    csv_power = f"{out_dir}/{model_name}-{system_name}-power.csv"

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
        file.write(f"    prompt: {prompt[:50].strip()}\n")

    load_model_event = threading.Event()
    power_readings = {
        "load model": [],
    }

    load_model_monitor_thread = threading.Thread(
        target=monitor_power_usage,
        args=(power_readings["load model"], load_model_event),
    )

    pre_mem = torch.mps.current_allocated_memory() / 1024**2
    pipe, tokenizer, model_load_runtime = load_model(
        model_name=args.hf_name,
        load_model_event=load_model_event,
        load_model_thread=load_model_monitor_thread,
    )
    model_mem = torch.mps.current_allocated_memory() / 1024**2  # in MB
    readings_load_model = process_data(
        power_readings=power_readings["load model"], event="load model"
    )
    df_energy = post_process_power_data(
        readings=readings_load_model, runtime_s=model_load_runtime
    )
    df_energy["Output Token Limit"] = num_tokens
    df_energy["Input Tokens"] = 0
    df_energy["Iteration"] = 0
    df_energy["Model Name"] = model_name
    df_energy["Number of GPUs"] = 1
    df_energy["Prompt"] = ""
    df_energy["Output Tokens"] = 0
    df_energy["Batch Size"] = batch_size
    df_energy["System"] = system_name
    df_energy["GPU Memory (MB)"] = [model_mem - pre_mem]
    df_energy.to_csv(
        f"{model_name}-{system_name}.csv", index=False, header=False, mode="a"
    )

    df_power = pd.DataFrame(readings_load_model)
    # print(df_power)
    df_power.to_csv(
        csv_power,
        index=False,
        header=False,
        mode="a",
    )

    output_token_lengths = [8, 16, 32, 64, 128, 256, 512]
    random.shuffle(output_token_lengths)
    for num_tokens in output_token_lengths:
        runtimes = []
        for i in range(10):
            dict_key = f"inference-{num_tokens}-{i}"
            power_readings[dict_key] = []
            inference_event = threading.Event()
            inference_monitor = threading.Thread(
                target=monitor_power_usage,
                args=(power_readings[dict_key], inference_event),
            )
            inference_start_time = time.time()
            llm_output = run_inference(
                pipe=pipe,
                num_tokens=num_tokens,
                prompt=prompt,
                inference_event=inference_event,
                inference_monitor=inference_monitor,
                batch_size=batch_size,
            )
            inference_runtime = time.time() - inference_start_time
            inference_mem = torch.mps.current_allocated_memory() / 1024**2
            print(llm_output)
            readings = process_data(power_readings[dict_key], dict_key)
            df_power = pd.DataFrame(readings)
            df_power.to_csv(
                csv_power,
                index=False,
                header=False,
                mode="a",
            )
            torch.mps.empty_cache()

            input_tokens = tokenizer.encode(prompt)
            num_input_tokens = len(input_tokens)
            output_tokens = tokenizer.encode(llm_output)
            num_output_tokens = len(output_tokens)
            df_energy_inference = post_process_power_data(readings, inference_runtime)
            df_energy_inference["Output Token Limit"] = num_tokens
            df_energy_inference["Input Tokens"] = num_input_tokens
            df_energy_inference["Iteration"] = i
            df_energy_inference["Model Name"] = model_name
            df_energy_inference["Number of GPUs"] = 1
            df_energy_inference["Prompt"] = prompt[:50].strip()
            df_energy_inference["Output Tokens"] = num_output_tokens - num_input_tokens
            df_energy_inference["Batch Size"] = batch_size
            df_energy_inference["System"] = system_name
            df_energy_inference["GPU Memory (MB)"] = inference_mem
            df_energy_inference.to_csv(
                f"{model_name}-{system_name}.csv", index=False, header=False, mode="a"
            )
            if i > 0:
                runtimes.append(inference_runtime)
            mean_runtime = np.mean(runtimes)
            std_err = stats.sem(runtimes)
            z_critical = stats.norm.ppf((1 + 0.95) / 2)
            ci_half_width = z_critical * std_err
            # Break if we have more than 5 samples and the confidence interval half-width is less than 0.5
            if i > 5 and ci_half_width < 0.5:
                break
