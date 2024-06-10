import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from datasets import load_dataset
from transformers import AutoTokenizer
import math
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# Coefficients for energy models
alpha_coeffs = {
    "Llama-2 (7B) AMD+A100": {
        "alpha_0": -3.894991,
        "alpha_1": 31.522735,
        "alpha_2": 0.042712,
    },
    "Llama-2 (7B) M1-Pro": {
        "alpha_0": -0.322916,
        "alpha_1": 4.644266,
        "alpha_2": 0.097094,
    },
}

# Coefficients for runtime models
beta_coeffs = {
    "Llama-2 (7B) AMD+A100": {
        "beta_0": -0.010021,
        "beta_1": 0.083515,
        "beta_2": 0.000107,
    },
    "Llama-2 (7B) M1-Pro": {
        "beta_0": -0.0150,
        "beta_1": 0.3664,
        "beta_2": 0.0072,
    },
    # Add other models as needed
}


def energy_cost(K, t_in, t_out):
    coeffs = alpha_coeffs[K]
    return (
        coeffs["alpha_0"] * t_in
        + coeffs["alpha_1"] * t_out
        + coeffs["alpha_2"] * t_in * t_out
    )


def runtime_cost(K, t_in, t_out):
    coeffs = beta_coeffs[K]
    return (
        coeffs["beta_0"] * t_in
        + coeffs["beta_1"] * t_out
        + coeffs["beta_2"] * t_in * t_out
    )


import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
datasets = {
    "Alpaca": load_dataset("vicgalle/alpaca-gpt4"),
    "GSM8K": load_dataset("gsm8k", "main"),
    "Python Codes 25K": load_dataset("flytech/python-codes-25k", split="train"),
}

results = []

for dataset_name, dataset in datasets.items():
    if dataset_name == "Alpaca":
        lengths_instructions = [
            len(tokenizer.encode(x)) for x in dataset["train"]["instruction"]
        ]
        lengths_inputs = [len(tokenizer.encode(x)) for x in dataset["train"]["input"]]
        lengths_outputs = [len(tokenizer.encode(x)) for x in dataset["train"]["output"]]
        lengths = [x + y for x, y in zip(lengths_instructions, lengths_inputs)]
        Q = zip(lengths, lengths_outputs)
    elif dataset_name == "GSM8K":
        lengths_inputs = [
            len(tokenizer.encode(x)) for x in dataset["train"]["question"]
        ]
        lengths_outputs = [len(tokenizer.encode(x)) for x in dataset["train"]["answer"]]
        Q = zip(lengths_inputs, lengths_outputs)
    elif dataset_name == "Python Codes 25K":
        lengths_inputs = [len(tokenizer.encode(x)) for x in dataset["text"]]
        lengths_outputs = [len(tokenizer.encode(x)) for x in dataset["output"]]
        Q = zip(lengths_inputs, lengths_outputs)

    Q = list(Q)
    thresholds = [i for i in range(1, 513)]

    for threshold in thresholds:
        total_energy_input = 0
        total_runtime_input = 0
        total_energy_output = 0
        total_runtime_output = 0
        total_energy_product = 0
        total_runtime_product = 0
        just_m1_pro_energy = 0
        just_swing_energy = 0
        just_m1_pro_runtime = 0
        just_swing_runtime = 0
        for t_in, t_out in Q:
            if t_in <= threshold:
                total_energy_input += energy_cost("Llama-2 (7B) M1-Pro", t_in, t_out)
                total_runtime_input += runtime_cost("Llama-2 (7B) M1-Pro", t_in, t_out)
            else:
                total_energy_input += energy_cost("Llama-2 (7B) AMD+A100", t_in, t_out)
                total_runtime_input += runtime_cost(
                    "Llama-2 (7B) AMD+A100", t_in, t_out
                )
            if t_out <= threshold:
                total_energy_output += energy_cost("Llama-2 (7B) M1-Pro", t_in, t_out)
                total_runtime_output += runtime_cost("Llama-2 (7B) M1-Pro", t_in, t_out)
            else:
                total_energy_output += energy_cost("Llama-2 (7B) AMD+A100", t_in, t_out)
                total_runtime_output += runtime_cost(
                    "Llama-2 (7B) AMD+A100", t_in, t_out
                )
            if t_in * t_out <= threshold * threshold:
                total_energy_product += energy_cost("Llama-2 (7B) M1-Pro", t_in, t_out)
                total_runtime_product += runtime_cost(
                    "Llama-2 (7B) M1-Pro", t_in, t_out
                )
            else:
                total_energy_product += energy_cost(
                    "Llama-2 (7B) AMD+A100", t_in, t_out
                )
                total_runtime_product += runtime_cost(
                    "Llama-2 (7B) AMD+A100", t_in, t_out
                )
            just_m1_pro_energy += energy_cost("Llama-2 (7B) M1-Pro", t_in, t_out)
            just_swing_energy += energy_cost("Llama-2 (7B) AMD+A100", t_in, t_out)
            just_m1_pro_runtime += runtime_cost("Llama-2 (7B) M1-Pro", t_in, t_out)
            just_swing_runtime += runtime_cost("Llama-2 (7B) AMD+A100", t_in, t_out)

        results.append(
            {
                "Dataset": dataset_name,
                "Threshold": threshold,
                "Total Energy (kWh)": total_energy_input / 3.6e6,
                "Runtime (s)": total_runtime_input,
                "Mean Runtime (s)": total_runtime_input / len(Q),
                "Method": "Input Threshold",
                "System Type": "Hybrid",
            }
        )
        results.append(
            {
                "Dataset": dataset_name,
                "Threshold": threshold,
                "Total Energy (kWh)": total_energy_output / 3.6e6,
                "Runtime (s)": total_runtime_output,
                "Mean Runtime (s)": total_runtime_output / len(Q),
                "Method": "Output Threshold",
                "System Type": "Hybrid",
            }
        )
        results.append(
            {
                "Dataset": dataset_name,
                "Threshold": threshold,
                "Total Energy (kWh)": total_energy_product / 3.6e6,
                "Runtime (s)": total_runtime_product,
                "Mean Runtime (s)": total_runtime_product / len(Q),
                "Method": "Product Threshold",
                "System Type": "Hybrid",
            }
        )
        results.append(
            {
                "Dataset": dataset_name,
                "Threshold": threshold,
                "Total Energy (kWh)": just_m1_pro_energy / 3.6e6,
                "Runtime (s)": just_m1_pro_runtime,
                "Mean Runtime (s)": just_m1_pro_runtime / len(Q),
                "Method": "M1-Pro",
                "System Type": "Full",
            }
        )
        results.append(
            {
                "Dataset": dataset_name,
                "Threshold": threshold,
                "Total Energy (kWh)": just_swing_energy / 3.6e6,
                "Runtime (s)": just_swing_runtime,
                "Mean Runtime (s)": just_swing_runtime / len(Q),
                "Method": "AMD+A100",
                "System Type": "Full",
            }
        )

results_df = pd.DataFrame(results)

sns.set(style="whitegrid", context="talk", font_scale=1.2, palette="colorblind")

# Plot for Total Energy
g_energy = sns.relplot(
    data=results_df,
    x="Threshold",
    y="Total Energy (kWh)",
    hue="Method",
    style="System Type",
    col="Dataset",
    col_wrap=3,
    kind="line",
    # marker="o",
    facet_kws={"sharey": False, "sharex": True},
)
g_energy.set_titles("{col_name}")
for ax in g_energy.axes.flat:
    ax.set_xscale("log", base=2)
    if ax.get_title() == "Alpaca":
        max_alpaca_energy = results_df[
            (results_df["Dataset"] == "Alpaca") & (results_df["Method"] == "AMD+A100")
        ]["Total Energy (kWh)"].max()
        min_alpaca_energy = results_df[
            (results_df["Dataset"] == "Alpaca") & (results_df["Method"] == "M1-Pro")
        ]["Total Energy (kWh)"].min()
        ax.set_ylim(math.floor(min_alpaca_energy) - 1, math.ceil(max_alpaca_energy) + 1)
        ax.set_yticks(
            np.linspace(math.floor(min_alpaca_energy), math.ceil(max_alpaca_energy), 5)
        )
    elif ax.get_title() == "GSM8K":
        max_gsm8k_energy = results_df[
            (results_df["Dataset"] == "GSM8K") & (results_df["Method"] == "AMD+A100")
        ]["Total Energy (kWh)"].max()
        min_gsm8k_energy = results_df[
            (results_df["Dataset"] == "GSM8K") & (results_df["Method"] == "M1-Pro")
        ]["Total Energy (kWh)"].min()
        ax.set_ylim(
            math.floor(min_gsm8k_energy) - 0.2, math.ceil(max_gsm8k_energy) + 0.2
        )
        ax.set_yticks(
            np.linspace(math.floor(min_gsm8k_energy), math.ceil(max_gsm8k_energy), 5)
        )
    elif ax.get_title() == "Python Codes 25K":
        max_python_codes_energy = results_df[
            (results_df["Dataset"] == "Python Codes 25K")
            & (results_df["Method"] == "AMD+A100")
        ]["Total Energy (kWh)"].max()
        min_python_codes_energy = results_df[
            (results_df["Dataset"] == "Python Codes 25K")
            & (results_df["Method"] == "M1-Pro")
        ]["Total Energy (kWh)"].min()
        ax.set_ylim(
            math.floor(min_python_codes_energy) - 0.5,
            math.ceil(max_python_codes_energy) + 0.5,
        )
        ax.set_yticks(
            np.linspace(
                math.floor(min_python_codes_energy),
                math.ceil(max_python_codes_energy),
                5,
            )
        )
plt.savefig("heterogeneous-threshold-energy.pdf", bbox_inches="tight")

# Plot for Runtime
sns.set(style="whitegrid", context="talk", font_scale=1.2, palette="colorblind")
g_runtime = sns.relplot(
    data=results_df,
    x="Threshold",
    y="Runtime (s)",
    hue="Method",
    style="System Type",
    col="Dataset",
    col_wrap=3,
    kind="line",
    # marker="o",
    facet_kws={"sharey": False, "sharex": True},
)
g_runtime.set_titles("{col_name}")
for ax in g_runtime.axes.flat:
    ax.set_xscale("log", base=2)
    if ax.get_title() == "Alpaca":
        min_alpaca_runtime = results_df[
            (results_df["Dataset"] == "Alpaca") & (results_df["Method"] == "AMD+A100")
        ]["Runtime (s)"].max()
        max_alpaca_runtime = results_df[
            (results_df["Dataset"] == "Alpaca") & (results_df["Method"] == "M1-Pro")
        ]["Runtime (s)"].min()
        ax.set_ylim(
            math.floor(min_alpaca_runtime) - 1, math.ceil(max_alpaca_runtime) + 1
        )
        ax.set_yticks(
            np.linspace(
                math.floor(min_alpaca_runtime), math.ceil(max_alpaca_runtime), 5
            )
        )
    elif ax.get_title() == "GSM8K":
        min_gsm8k_runtime = results_df[
            (results_df["Dataset"] == "GSM8K") & (results_df["Method"] == "AMD+A100")
        ]["Runtime (s)"].max()
        max_gsm8k_runtime = results_df[
            (results_df["Dataset"] == "GSM8K") & (results_df["Method"] == "M1-Pro")
        ]["Runtime (s)"].min()
        ax.set_ylim(math.floor(min_gsm8k_runtime), math.ceil(max_gsm8k_runtime))
        ax.set_yticks(
            np.linspace(math.floor(min_gsm8k_runtime), math.ceil(max_gsm8k_runtime), 5)
        )
    elif ax.get_title() == "Python Codes 25K":
        min_python_codes_runtime = results_df[
            (results_df["Dataset"] == "Python Codes 25K")
            & (results_df["Method"] == "AMD+A100")
        ]["Runtime (s)"].max()
        max_python_codes_runtime = results_df[
            (results_df["Dataset"] == "Python Codes 25K")
            & (results_df["Method"] == "M1-Pro")
        ]["Runtime (s)"].min()
        ax.set_ylim(
            math.floor(min_python_codes_runtime) - 1,
            math.ceil(max_python_codes_runtime) + 1,
        )
        ax.set_yticks(
            np.linspace(
                math.floor(min_python_codes_runtime),
                math.ceil(max_python_codes_runtime),
                5,
            )
        )
plt.savefig("heterogeneous-threshold-runtime.pdf", bbox_inches="tight")


# Plot for Total Energy
g_energy = sns.relplot(
    data=results_df,
    x="Threshold",
    y="Total Energy (kWh)",
    hue="Method",
    style="System Type",
    col="Dataset",
    col_wrap=3,
    linewidth=3,
    kind="line",
    # marker="o",
    facet_kws={"sharey": False, "sharex": True},
)
g_energy.set_titles("{col_name}")
for ax in g_energy.axes.flat:
    if ax.get_title() == "Alpaca":
        ax.set_ylim(0, 80)
    elif ax.get_title() == "GSM8K":
        ax.set_ylim(0, 10)
    elif ax.get_title() == "Python Codes 25K":
        ax.set_ylim(0, 60)

plt.savefig("heterogeneous-threshold-energy-linear.pdf", bbox_inches="tight")

# Plot for Runtime
sns.set(style="whitegrid", context="talk", font_scale=1.5, palette="colorblind")
g_runtime = sns.relplot(
    data=results_df,
    x="Threshold",
    y="Mean Runtime (s)",
    hue="Method",
    style="System Type",
    col="Dataset",
    col_wrap=3,
    kind="line",
    linewidth=3,
    # marker="o",
    facet_kws={"sharey": False, "sharex": True},
)
g_runtime.set_titles("{col_name}")
for ax in g_runtime.axes.flat:
    if ax.get_title() == "Alpaca":
        ax.set_ylim(0, 100)
    elif ax.get_title() == "GSM8K":
        ax.set_ylim(0, 120)
    elif ax.get_title() == "Python Codes 25K":
        ax.set_ylim(0, 250)
plt.savefig("heterogeneous-threshold-runtime-linear.pdf", bbox_inches="tight")
