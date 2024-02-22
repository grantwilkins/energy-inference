import torch
from pyJoules.energy_meter import EnergyContext
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from pyJoules.handler.csv_handler import CSVHandler
import argparse
from time import sleep

parser = argparse.ArgumentParser()

parser.add_argument("--num_tokens", type=int, default=200)
parser.add_argument("--model_name", type=str, default="pyjoules")

args = parser.parse_args()


num_gpus = torch.cuda.device_count()
stats_name = args.model_name
csv_handle = CSVHandler(f"{stats_name}-{num_gpus}.csv")
num_tokens = args.num_tokens
# domains = NvididaGPUDomain0 if num_gpus == 1 else [i for i in range(num_gpus)]

with EnergyContext(
    handler=csv_handle,
    domains=[NvidiaGPUDomain(i) for i in range(num_gpus)],
    start_tag="tokenizer",
) as ctx:
    sleep(5)
    ctx.record(tag="model load")
    sleep(5)
    ctx.record(tag="pipeline load")
    sleep(5)

csv_handle.save_data()
