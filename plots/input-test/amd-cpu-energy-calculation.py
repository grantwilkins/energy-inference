import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Load the data
home_path = "/Users/grantwilkins/energy-inference/"
path_to_amd_results_falcon = """cuda/falcon-7b/2024-04-02/07-36-20/AMDuProf-python3-Timechart_Apr-02-2024_07-36-21"""
path_to_amd_results_mistral = """cuda/mistral-7b/2024-03-28/11-19-45/AMDuProf-python3-Timechart_Mar-28-2024_11-19-46"""
path_to_amd_results_llama = """cuda/llama2-7b/2024-03-28/11-15-46/AMDuProf-python3-Timechart_Mar-28-2024_11-15-48"""

df_falcon_amd = pd.read_csv(
    os.path.join(home_path, path_to_amd_results_falcon, "timechart.csv"), skiprows=150
)

df_mistral_amd = pd.read_csv(
    os.path.join(home_path, path_to_amd_results_mistral, "timechart.csv"), skiprows=150
)

df_llama_amd = pd.read_csv(
    os.path.join(home_path, path_to_amd_results_llama, "timechart.csv"), skiprows=150
)

falcon_cpu_cores = [73, 74]
mistral_cpu_cores = [47, 47]
llama_cpu_cores = [35, 39]

time_width = 0.1

total_time = df_falcon_amd["RecordId"].iloc[-1] * time_width

falcon_cpu_energy = (
    df_falcon_amd[f"core{falcon_cpu_cores[0]}-power"].sum()
    + df_falcon_amd[f"core{falcon_cpu_cores[1]}-power"].sum()
) * time_width
mistral_cpu_energy = (
    df_mistral_amd[f"core{falcon_cpu_cores[0]}-power"].sum()
    + df_mistral_amd[f"core{falcon_cpu_cores[1]}-power"].sum()
) * time_width
llama_cpu_energy = (
    df_llama_amd[f"core{llama_cpu_cores[0]}-power"].sum()
    + df_llama_amd[f"core{llama_cpu_cores[1]}-power"].sum()
) * time_width

df_falcon_stats = pd.read_csv("falcon-7b-argonne-swing-1.csv")
df_mistral_stats = pd.read_csv("Mistral-7B-v0.1-argonne-swing-1.csv")
df_llama_stats = pd.read_csv("Llama-2-7b-chat-hf-argonne-swing-1.csv")

df_falcon_stats["CPU Energy (J)"] = (
    falcon_cpu_energy * df_falcon_stats["Runtime (s)"] / total_time
)
df_mistral_stats["CPU Energy (J)"] = (
    mistral_cpu_energy * df_mistral_stats["Runtime (s)"]
) / total_time
df_llama_stats["CPU Energy (J)"] = (
    llama_cpu_energy * df_llama_stats["Runtime (s)"] / total_time
)

df_falcon_stats.to_csv(
    "falcon-7b-argonne-swing-1.csv", sep=",", index=False, header=True
)
df_mistral_stats.to_csv(
    "Mistral-7B-v0.1-argonne-swing-1.csv", sep=",", index=False, header=True
)
df_llama_stats.to_csv(
    "Llama-2-7b-chat-hf-argonne-swing-1.csv", sep=",", index=False, header=True
)
