import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Load the data
home_path = "/Users/grantwilkins/energy-inference/"
path_to_amd_results_falcon = """plots/parameter-scale-output-test/raw_stats/falcon-40b/2024-04-18/10-13-21/AMDuProf-python3-Timechart_Apr-18-2024_10-13-21"""
path_to_amd_results_mixtral = """plots/parameter-scale-output-test/raw_stats/mixtral-8x7b/2024-04-10/09-42-22/AMDuProf-python3-Timechart_Apr-10-2024_09-42-23"""
path_to_amd_results_llama_13b = """plots/parameter-scale-output-test/raw_stats/llama2-13b/2024-04-21/18-08-10/AMDuProf-python3-Timechart_Apr-21-2024_18-08-12"""
path_to_amd_results_llama_70b = """plots/parameter-scale-output-test/raw_stats/llama2-70b/2024-04-12/09-27-17/AMDuProf-python3-Timechart_Apr-12-2024_09-27-18"""

df_falcon_amd = pd.read_csv(
    os.path.join(home_path, path_to_amd_results_falcon, "timechart.csv"), skiprows=150
)

df_mixtral_amd = pd.read_csv(
    os.path.join(home_path, path_to_amd_results_mixtral, "timechart.csv"), skiprows=150
)

df_llama_13b_amd = pd.read_csv(
    os.path.join(home_path, path_to_amd_results_llama_13b, "timechart.csv"),
    skiprows=150,
)

df_llama_70b_amd = pd.read_csv(
    os.path.join(home_path, path_to_amd_results_llama_13b, "timechart.csv"),
    skiprows=150,
)

falcon_cpu_cores = [x for x in range(32, 47)]
mixtral_cpu_cores = [x for x in range(32, 47)]
llama_13b_cpu_cores = [x for x in range(0, 15)]
llama_70b_cpu_cores = [x for x in range(32, 63)]

time_width = 0.1

total_time_falcon = df_falcon_amd["RecordId"].iloc[-1] * time_width
total_time_mixtral = df_mixtral_amd["RecordId"].iloc[-1] * time_width
total_time_llama_13b = df_llama_13b_amd["RecordId"].iloc[-1] * time_width
total_time_llama_70b = df_llama_70b_amd["RecordId"].iloc[-1] * time_width


def calculate_cpu_energy(df, cpu_cores):
    return sum(df[f"core{core}-power"].sum() for core in cpu_cores) * time_width


falcon_cpu_energy = calculate_cpu_energy(df_falcon_amd, falcon_cpu_cores)
mixtral_cpu_energy = calculate_cpu_energy(df_mixtral_amd, mixtral_cpu_cores)
llama_13b_cpu_energy = calculate_cpu_energy(df_llama_13b_amd, llama_13b_cpu_cores)
llama_70b_cpu_energy = calculate_cpu_energy(df_llama_70b_amd, llama_70b_cpu_cores)

df_falcon_stats = pd.read_csv("falcon-40b-argonne-swing-3.csv")
df_mixtral_stats = pd.read_csv("Mixtral-8x7B-v0.1-argonne-swing-3.csv")
df_llama_13b_stats = pd.read_csv("Llama-2-13b-chat-hf-argonne-swing-1.csv")
df_llama_70b_stats = pd.read_csv("Llama-2-70b-chat-hf-argonne-swing-4.csv")

df_falcon_stats["CPU Energy (J)"] = (
    falcon_cpu_energy * df_falcon_stats["Runtime (s)"] / total_time_falcon
)
df_mixtral_stats["CPU Energy (J)"] = (
    mixtral_cpu_energy * df_mixtral_stats["Runtime (s)"]
) / total_time_mixtral
df_llama_13b_stats["CPU Energy (J)"] = (
    llama_13b_cpu_energy * df_llama_13b_stats["Runtime (s)"] / total_time_llama_13b
)
df_llama_70b_stats["CPU Energy (J)"] = (
    llama_70b_cpu_energy * df_llama_70b_stats["Runtime (s)"] / total_time_llama_70b
)

df_falcon_stats["GPU Energy (J)"] = (
    df_falcon_stats["GPU-0 Energy (mJ)"]
    + df_falcon_stats["GPU-1 Energy (mJ)"]
    + df_falcon_stats["GPU-2 Energy (mJ)"]
) / 1e3
df_mixtral_stats["GPU Energy (J)"] = (
    df_mixtral_stats["GPU-0 Energy (mJ)"]
    + df_mixtral_stats["GPU-1 Energy (mJ)"]
    + df_mixtral_stats["GPU-2 Energy (mJ)"]
) / 1e3
df_llama_13b_stats["GPU Energy (J)"] = df_llama_13b_stats["GPU-0 Energy (mJ)"] / 1e3
df_llama_70b_stats["GPU Energy (J)"] = (
    df_llama_70b_stats["GPU-0 Energy (mJ)"]
    + df_llama_70b_stats["GPU-1 Energy (mJ)"]
    + df_llama_70b_stats["GPU-2 Energy (mJ)"]
    + df_llama_70b_stats["GPU-3 Energy (mJ)"]
) / 1e3

df_falcon_stats.to_csv(
    "falcon-40b-argonne-swing-3.csv", sep=",", index=False, header=True
)
df_mixtral_stats.to_csv(
    "Mixtral-8x7B-v0.1-argonne-swing-3.csv", sep=",", index=False, header=True
)
df_llama_13b_stats.to_csv(
    "Llama-2-13b-chat-hf-argonne-swing-1.csv", sep=",", index=False, header=True
)
df_llama_70b_stats.to_csv(
    "Llama-2-70b-chat-hf-argonne-swing-4.csv", sep=",", index=False, header=True
)
