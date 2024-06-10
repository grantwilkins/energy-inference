import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Load the data
home_path = "/Users/grantwilkins/energy-inference/"
path_to_amd_results_falcon = """plots/output-test/raw_stats/falcon-7b/2024-03-29/13-02-36/AMDuProf-python3-Timechart_Mar-29-2024_13-02-38"""
path_to_amd_results_mistral = """plots/output-test/raw_stats/llama2-7b/2024-04-03/06-26-22/AMDuProf-python3-Timechart_Apr-03-2024_06-26-23"""
path_to_amd_results_llama = """plots/output-test/raw_stats/mistral-7b/2024-04-03/06-26-50/AMDuProf-python3-Timechart_Apr-03-2024_06-26-51"""

df_falcon_amd = pd.read_csv(
    os.path.join(home_path, path_to_amd_results_falcon, "timechart.csv"), skiprows=150
)

df_mistral_amd = pd.read_csv(
    os.path.join(home_path, path_to_amd_results_mistral, "timechart.csv"), skiprows=150
)

df_llama_amd = pd.read_csv(
    os.path.join(home_path, path_to_amd_results_llama, "timechart.csv"), skiprows=150
)

falcon_cpu_cores = [x for x in range(16, 33)]
mistral_cpu_cores = [x for x in range(0, 17)]
llama_cpu_cores = [x for x in range(64, 81)]

time_width = 0.1

total_time_falcon = df_falcon_amd["RecordId"].iloc[-1] * time_width
total_time_mistral = df_mistral_amd["RecordId"].iloc[-1] * time_width
total_time_llama = df_llama_amd["RecordId"].iloc[-1] * time_width


def calculate_cpu_energy(df, cpu_cores):
    return sum(df[f"core{core}-power"].sum() for core in cpu_cores) * time_width


falcon_cpu_energy = calculate_cpu_energy(df_falcon_amd, falcon_cpu_cores)
mistral_cpu_energy = calculate_cpu_energy(df_mistral_amd, mistral_cpu_cores)
llama_cpu_energy = calculate_cpu_energy(df_llama_amd, llama_cpu_cores)

df_falcon_stats = pd.read_csv("falcon-7b-argonne-swing-1.csv")
df_mistral_stats = pd.read_csv("Mistral-7B-v0.1-argonne-swing-1.csv")
df_llama_stats = pd.read_csv("Llama-2-7b-chat-hf-argonne-swing-1.csv")

df_falcon_stats["CPU Energy (J)"] = (
    falcon_cpu_energy * df_falcon_stats["Runtime (s)"] / total_time_falcon
)
df_mistral_stats["CPU Energy (J)"] = (
    mistral_cpu_energy * df_mistral_stats["Runtime (s)"]
) / total_time_mistral
df_llama_stats["CPU Energy (J)"] = (
    llama_cpu_energy * df_llama_stats["Runtime (s)"] / total_time_llama
)

df_falcon_stats["GPU Energy (J)"] = df_falcon_stats["GPU Energy (mJ)"] / 1e3
df_mistral_stats["GPU Energy (J)"] = df_mistral_stats["GPU Energy (mJ)"] / 1e3
df_llama_stats["GPU Energy (J)"] = df_llama_stats["GPU Energy (mJ)"] / 1e3

df_falcon_stats.to_csv(
    "falcon-7b-argonne-swing-1.csv", sep=",", index=False, header=True
)
df_mistral_stats.to_csv(
    "Mistral-7B-v0.1-argonne-swing-1.csv", sep=",", index=False, header=True
)
df_llama_stats.to_csv(
    "Llama-2-7b-chat-hf-argonne-swing-1.csv", sep=",", index=False, header=True
)
