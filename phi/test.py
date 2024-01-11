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
            print(split_line)
            power_usage[split_line[0]] = split_line[-1]
        if cpu_match:
            power_usage["cpu"] = int(cpu_match.group(1))
        if gpu_match:
            power_usage["gpu"] = int(gpu_match.group(1))
        if power_usage.keys() == {"cpu", "gpu", "python3.11", "ALL_TASKS"}:
            readings.append(power_usage)
            power_usage = {}
    return readings


def run_inference():
    print("Starting inference...")
    time.sleep(5)  # Replace with the actual duration of your inference task
    print("Inference completed.")


# Flag for monitoring control and storage for power readings
stop_monitoring = threading.Event()
power_readings = []

# Start the power monitoring in a separate thread
monitor_thread = threading.Thread(
    target=monitor_power_usage, args=(power_readings, stop_monitoring)
)
monitor_thread.start()

# Run the inference process
run_inference()

# Signal to stop the monitoring thread and wait for it to finish
stop_monitoring.set()
monitor_thread.join()
print("Powermetrics monitoring stopped.")

# Process the collected data
readings = process_data(power_readings)
