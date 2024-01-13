from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import subprocess
import threading
import re
import time


def monitor_power_usage(power_readings, stop_monitoring):
    cmd = ["sudo", "powermetrics", "--show-process-energy", "-i", "500"]
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True) as process:
        while not stop_monitoring.is_set():
            line = process.stdout.readline()
            if not line:
                break
            power_readings.append(line)
    print("Powermetrics process ended.")


def process_data(power_readings):
    cpu_pattern = r"CPU Power:\s+(\d+)\s+mW"
    gpu_pattern = r"GPU Power:\s+(\d+)\s+mW"
    keywords = ["python3.11", "ALL_TASKS"]
    readings = []
    power_usage = {}
    for line in power_readings:
        if not line:
            break
        cpu_match = re.search(cpu_pattern, line)
        gpu_match = re.search(gpu_pattern, line)
        if any(line.startswith(keyword) for keyword in keywords):
            split_line = line.split()
            power_usage[split_line[0]] = split_line[-1]
        if cpu_match:
            power_usage["cpu"] = int(cpu_match.group(1))
        if gpu_match:
            power_usage["gpu"] = int(gpu_match.group(1))
        if power_usage.keys() == {"cpu", "gpu", "python3.11", "ALL_TASKS"}:
            readings.append(power_usage)
            power_usage = {}
    return readings


model_name = "microsoft/phi-2"
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer("The sky is blue because", return_tensors="pt")
model.to("mps")
inputs = inputs.to("mps")
# Flag for monitoring control and storage for power readings
stop_monitoring = threading.Event()
power_readings = []

# Start the power monitoring in a separate thread
monitor_thread = threading.Thread(
    target=monitor_power_usage, args=(power_readings, stop_monitoring)
)
monitor_thread.start()

start = time.time()
outputs = model.generate(**inputs, max_length=200)
text = tokenizer.batch_decode(outputs)[0]
end = time.time() - start

stop_monitoring.set()
monitor_thread.join()
print("Powermetrics monitoring stopped.")

print("total tokens count: " + str(outputs.size(1)))


# Process the collected data
readings = process_data(power_readings)
print(readings)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

time = np.arange(0, end, end / len(readings))

df = pd.DataFrame(readings)
df["time"] = time

# df["gpu_energy"] = df["gpu"].cumsum() * (end / len(readings))
# print(df["gpu_energy"])

df.plot(x="time", y=["python3.11", "gpu"])
plt.show()
