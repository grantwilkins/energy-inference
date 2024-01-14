from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import subprocess
import threading
import re
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import sleep


def monitor_power_usage(power_readings, stop_monitoring):
    cmd = ["sudo", "powermetrics", "--show-process-energy", "-i", "200"]
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True) as process:
        while not stop_monitoring.is_set():
            line = process.stdout.readline()
            if not line:
                break
            power_readings.append(line)
    print("Powermetrics process ended.")


def process_data(power_readings, event):
    cpu_pattern = r"CPU Power:\s+(\d+)\s+mW"
    gpu_pattern = r"GPU Power:\s+(\d+)\s+mW"
    keywords = ["python3.11", "ALL_TASKS"]
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
            "python3.11",
            "ALL_TASKS",
            "event",
            "timestamp",
        }:
            readings.append(power_usage)
            power_usage = {"event": event}
    return readings


num_tokens = 200
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer_event = threading.Event()
pipeline_load_event = threading.Event()
inference_event = threading.Event()
power_readings = {
    "tokenizer": [],
    "pipeline load": [],
    "inference": [],
}
# Start the power monitoring in a separate thread
tokenizer_monitor = threading.Thread(
    target=monitor_power_usage, args=(power_readings["tokenizer"], tokenizer_event)
)
pipeline_load_monitor = threading.Thread(
    target=monitor_power_usage,
    args=(power_readings["pipeline load"], pipeline_load_event),
)
inference_monitor = threading.Thread(
    target=monitor_power_usage, args=(power_readings["inference"], inference_event)
)

tokenizer_monitor.start()
sleep(2)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer_event.set()
tokenizer_monitor.join()

pipeline_load_monitor.start()
sleep(2)
pipe = pipeline(
    "text-generation",
    model=model_name,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)
pipeline_load_event.set()
pipeline_load_monitor.join()


inference_monitor.start()
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
inference_event.set()
inference_monitor.join()


print(sequences[0]["generated_text"])

# Process the collected data
readings = process_data(power_readings["tokenizer"], "tokenizer")
readings.extend(process_data(power_readings["pipeline load"], "pipeline load"))
readings.extend(process_data(power_readings["inference"], "inference"))

df_power = pd.DataFrame(readings)

df_power.to_csv("llama2-mps.csv", index=False, header=True, mode="a")
