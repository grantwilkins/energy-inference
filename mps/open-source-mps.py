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


def tokenizer_model_pipeline(
    model_name: str,
    tokenizer_event: threading.Event,
    tokenizer_thread: threading.Thread,
    model_load_event: threading.Event,
    model_load_thread: threading.Thread,
    pipeline_load_event: threading.Event,
    pipeline_load_thread: threading.Thread,
) -> tuple[Pipeline, AutoTokenizer, tuple[float, float, float]]:
    tokenizer_thread.start()
    sleep(2)
    tokenizer_start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer_time = time.time() - tokenizer_start_time
    tokenizer_event.set()
    tokenizer_thread.join()

    model_load_thread.start()
    sleep(2)
    model_load_start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model_load_time = time.time() - model_load_start_time
    model_load_event.set()
    model_load_thread.join()

    pipeline_load_thread.start()
    sleep(2)
    pipeline_start_time = time.time()
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    pipeline_load_time = time.time() - pipeline_start_time
    pipeline_load_event.set()
    pipeline_load_thread.join()

    return (
        pipe,
        tokenizer,
        (tokenizer_time, model_load_time, pipeline_load_time),
    )


def run_inference(
    pipe: Pipeline,
    num_tokens: int,
    prompt: str,
    inference_event: threading.Event,
    inference_monitor: threading.Thread,
) -> str:
    inference_monitor.start()
    sleep(2)
    sequences = pipe(
        prompt,
        do_sample=True,
        max_new_tokens=num_tokens,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
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
        total_cpu_energy += reading["cpu"] * reading["timestamp"] / 1000 / 1000
    df["CPU Energy (J)"] = [total_cpu_energy]
    total_gpu_energy = 0
    for reading in readings:
        total_gpu_energy += reading["gpu"] * reading["timestamp"] / 1000 / 1000
    df["GPU Energy (J)"] = [total_gpu_energy]
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_tokens", type=int, default=256)
    parser.add_argument("--hf_name", type=str, default="mistralai/Mistral-7B-v0.1")
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
    prompts = {
        "A": "What is the capital of France?",
        "B": "Can you explain the difference between a simile and a metaphor? Provide an example of each.",
        "C": "What are some effective strategies for managing stress and maintaining good mental health during challenging times, such as a pandemic or a personal crisis?",
        "D": "Imagine you are a travel guide. Can you recommend a 7-day itinerary for a trip to Japan, including must-visit destinations, cultural experiences, and local cuisine? Provide a brief description of each day's activities and how they showcase the best of Japan.",
        "E": "As an AI language model, you have the ability to process and generate human-like text. Can you discuss the potential implications of advanced AI systems like yourself on various industries, such as healthcare, education, and creative fields? Consider the benefits, challenges, and ethical considerations surrounding the integration of AI in these sectors. Provide specific examples to support your analysis.",
    }

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
        for idx, prompt in prompts.items():
            file.write(f"    prompt-{idx}: {prompt}\n")

    tokenizer_event = threading.Event()
    model_load_event = threading.Event()
    pipeline_load_event = threading.Event()
    power_readings = {
        "tokenizer": [],
        "model load": [],
        "pipeline load": [],
    }

    tokenizer_monitor = threading.Thread(
        target=monitor_power_usage, args=(power_readings["tokenizer"], tokenizer_event)
    )
    model_load_monitor = threading.Thread(
        target=monitor_power_usage,
        args=(power_readings["model load"], model_load_event),
    )
    pipeline_load_monitor = threading.Thread(
        target=monitor_power_usage,
        args=(power_readings["pipeline load"], pipeline_load_event),
    )

    pre_mem = torch.mps.current_allocated_memory() / 1024**2
    pipe, tokenizer, (tokenizer_runtime, model_load_runtime, pipeline_runtime) = (
        tokenizer_model_pipeline(
            args.hf_name,
            tokenizer_event,
            tokenizer_monitor,
            model_load_event,
            model_load_monitor,
            pipeline_load_event,
            pipeline_load_monitor,
        )
    )
    model_mem = torch.mps.current_allocated_memory() / 1024**2  # in MB
    readings_tokenizer = process_data(power_readings["tokenizer"], "tokenizer")

    df_energy_tokenizer = post_process_power_data(readings_tokenizer, tokenizer_runtime)

    readings_model_load = process_data(power_readings["model load"], "model load")
    readings_tokenizer.extend(readings_model_load)

    df_energy_model_load = post_process_power_data(
        readings_model_load, model_load_runtime
    )

    readings_pipeline_load = process_data(
        power_readings["pipeline load"], "pipeline load"
    )
    readings_tokenizer.extend(
        readings_pipeline_load,
    )
    df_energy_pipeline_load = post_process_power_data(
        readings_pipeline_load, pipeline_runtime
    )
    df_energy = pd.concat(
        [df_energy_tokenizer, df_energy_model_load, df_energy_pipeline_load],
        ignore_index=True,
    )
    df_energy["Output Token Limit"] = num_tokens
    df_energy["Input Tokens"] = 0
    df_energy["Iteration"] = 0
    df_energy["Model Name"] = model_name
    df_energy["Number of GPUs"] = 1
    df_energy["Prompt"] = "startup"
    df_energy["Output Tokens"] = 0
    df_energy["Batch Size"] = batch_size
    df_energy["System"] = system_name
    df_energy["GPU Memory (MB)"] = [pre_mem, model_mem, model_mem]
    df_energy.to_csv(
        f"{model_name}-{system_name}.csv", index=False, header=False, mode="a"
    )

    df_power = pd.DataFrame(readings_tokenizer)
    # print(df_power)
    df_power.to_csv(
        csv_power,
        index=False,
        header=False,
        mode="a",
    )

    for idx, prompt in prompts.items():
        for i in range(5):
            dict_key = f"inference-{idx}-{i}"
            power_readings[dict_key] = []
            inference_event = threading.Event()
            inference_monitor = threading.Thread(
                target=monitor_power_usage,
                args=(power_readings[dict_key], inference_event),
            )
            inference_start_time = time.time()
            llm_output = run_inference(
                pipe, num_tokens, prompt, inference_event, inference_monitor
            )
            inference_runtime = time.time() - inference_start_time
            inference_mem = torch.mps.current_allocated_memory() / 1024**2
            readings = process_data(power_readings[dict_key], dict_key)
            df_power = pd.DataFrame(readings)
            df_power.to_csv(
                csv_power,
                index=False,
                header=False,
                mode="a",
            )

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
            df_energy_inference["Prompt"] = prompt
            df_energy_inference["Output Tokens"] = num_output_tokens
            df_energy_inference["Batch Size"] = batch_size
            df_energy_inference["System"] = system_name
            df_energy_inference["GPU Memory (MB)"] = inference_mem
            df_energy_inference.to_csv(
                f"{model_name}-{system_name}.csv", index=False, header=False, mode="a"
            )
