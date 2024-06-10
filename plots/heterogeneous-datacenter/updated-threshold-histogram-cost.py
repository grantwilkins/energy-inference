import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from datasets import load_dataset
from transformers import AutoTokenizer
import matplotlib
import pulp as lp
import math
import random

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

np.random.seed(42)

models = ["Llama-2 (7B) AMD+A100", "Llama-2 (7B) M1-Pro"]

gamma_K = {"Llama-2 (7B) AMD+A100": 0.75, "Llama-2 (7B) M1-Pro": 0.25}

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


min_values = {}
max_values = {}
max_input_tokens = 512
max_output_tokens = 512
for model in models:
    max_runtime = runtime_cost(model, max_input_tokens, max_output_tokens)
    max_energy = energy_cost(model, max_input_tokens, max_output_tokens)
    max_values[model] = {"max_runtime": max_runtime, "max_energy": max_energy}
    min_values[model] = {"min_runtime": 0, "min_energy": 0}


def normalize(value, min_value, max_value):
    return (
        (value - min_value) / (max_value - min_value) if max_value != min_value else 0
    )


def normalized_energy_cost(K, t_in, t_out):
    energy_min = min_values[K]["min_energy"]
    energy_max = max_values[K]["max_energy"]
    return normalize(energy_cost(K, t_in, t_out), energy_min, energy_max)


def normalized_runtime_cost(K, t_in, t_out):
    runtime_min = min_values[K]["min_runtime"]
    runtime_max = max_values[K]["max_runtime"]
    return normalize(runtime_cost(K, t_in, t_out), runtime_min, runtime_max)


def calculate_metrics(assignments):
    total_energy = 0
    total_runtime = 0
    for model, queries in assignments.items():
        for t_in, t_out in queries:
            total_energy += energy_cost(model, t_in, t_out)
            total_runtime += runtime_cost(model, t_in, t_out)
    return total_energy, total_runtime


def cost(zeta, K, t_in, t_out):
    return zeta * normalized_energy_cost(K, t_in, t_out) + (
        1 - zeta
    ) * normalized_runtime_cost(K, t_in, t_out)


def dynamic_normalized_cost(cost_func, K, t_in, t_out, current_values):
    min_val = min(current_values) if current_values else 0
    max_val = max(current_values) if current_values else 1
    return normalize(cost_func(K, t_in, t_out), min_val, max_val)


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
    num_queries = 100
    Q = Q[:num_queries]
    x = {
        (k, i): lp.LpVariable(f"x_{k}_{i}", cat="Binary")
        for k in models
        for i in range(num_queries)
    }

    zeta_values = np.arange(0, 1.1, 0.1)
    for zeta in zeta_values:
        current_energy_costs = [
            energy_cost(k, t_in, t_out) for k in models for t_in, t_out in Q
        ]
        current_runtime_costs = [
            runtime_cost(k, t_in, t_out) for k in models for t_in, t_out in Q
        ]
        problem = lp.LpProblem("Cost Optimization", lp.LpMinimize)
        problem += (
            lp.lpSum(
                [
                    zeta
                    * (
                        dynamic_normalized_cost(
                            energy_cost, k, t_in, t_out, current_energy_costs
                        )
                        * x[(k, i)]
                    )
                    + (1 - zeta)
                    * dynamic_normalized_cost(
                        runtime_cost, k, t_in, t_out, current_runtime_costs
                    )
                    * x[(k, i)]
                    for k in models
                    for i in range(num_queries)
                    for t_in, t_out in Q
                ]
            ),
            "OptimizationObjective",
        )

        for i in range(num_queries):
            problem += lp.lpSum(x[(k, i)] for k in models) == 1, f"Query_Assignment_{i}"

        # for k in models:
        #     max_limit = num_queries * gamma_K[k] * 2.0
        #     problem += (
        #         lp.lpSum(x[(k, i)] for i in range(num_queries)) <= max_limit,
        #         f"Max_Workload_{k}",
        #     )

        for k in models:
            min_queries = max(1, num_queries * gamma_K[k] * 0.25)
            problem += (
                lp.lpSum(x[(k, i)] for i in range(num_queries)) >= min_queries,
                f"Min_Workload_{k}",
            )

        problem.solve()

        round_robin_assignments = {}
        for i in range(num_queries):
            model_index = i % 2
            model_name = models[model_index]
            if model_name not in round_robin_assignments:
                round_robin_assignments[model_name] = []
            round_robin_assignments[model_name].append(Q[i])

        # Random Routing
        random_assignments = {}
        for i in range(num_queries):
            model_name = random.choice(models)
            if model_name not in random_assignments:
                random_assignments[model_name] = []
            random_assignments[model_name].append(Q[i])

        total_energy_rr, total_runtime_rr = calculate_metrics(round_robin_assignments)
        total_energy_random, total_runtime_random = calculate_metrics(
            random_assignments
        )
        print(f"Round Robin: Total Energy {total_energy_rr:.2f}")
        print(f"Random: Total Energy {total_energy_random:.2f}")

        total_energy = 0
        total_runtime = 0
        total_energy_model = {k: 0 for k in models}
        for k in models:
            assigned_queries = [
                (Q[i][0], Q[i][1])
                for i in range(num_queries)
                if x[(k, i)].varValue == 1
            ]
            for t_in, t_out in assigned_queries:
                total_energy += energy_cost(k, t_in, t_out)
                total_runtime += runtime_cost(k, t_in, t_out)
                total_energy_model[k] += energy_cost(k, t_in, t_out)
        print(f"Model {k}: Total Energy {total_energy:.2f}")
        results.append(
            {
                "Dataset": dataset_name,
                "Method": "Offline Solution",
                "Zeta": zeta,
                "Cost": problem.objective.value(),
                "System Type": "Hybrid",
                "Total Energy (J)": total_energy,
                "Runtime (s)": total_runtime,
            }
        )
        # results.append(
        #     {
        #         "Dataset": dataset_name,
        #         "Method": "Offline",
        #         "Zeta": zeta,
        #         "Cost": problem.objective.value(),
        #         "System Type": "AMD+A100",
        #         "Total Energy (J)": total_energy_model["Llama-2 (7B) AMD+A100"],
        #         "Runtime (s)": total_runtime,
        #     }
        # )
        # results.append(
        #     {
        #         "Dataset": dataset_name,
        #         "Method": "Offline",
        #         "Zeta": zeta,
        #         "Cost": problem.objective.value(),
        #         "System Type": "M1-Pro",
        #         "Total Energy (J)": total_energy_model["Llama-2 (7B) M1-Pro"],
        #         "Runtime (s)": total_runtime,
        #     }
        # )
        for k in models:
            total_energy = 0
            total_runtime = 0
            for t_in, t_out in Q:
                total_energy += energy_cost(k, t_in, t_out)
                total_runtime += runtime_cost(k, t_in, t_out)
            results.append(
                {
                    "Dataset": dataset_name,
                    "Method": f"100% {k.split(' ')[-1]}",
                    "Zeta": zeta,
                    "Cost": cost(zeta, k, t_in, t_out),
                    "System Type": k,
                    "Total Energy (J)": total_energy,
                    "Runtime (s)": total_runtime,
                }
            )
        results.append(
            {
                "Dataset": dataset_name,
                "Method": "Round Robin",
                "Zeta": zeta,
                "Cost": 0,
                "System Type": "Hybrid",
                "Total Energy (J)": total_energy_rr,
                "Runtime (s)": total_runtime_rr,
            }
        )
        results.append(
            {
                "Dataset": dataset_name,
                "Method": "Random",
                "Zeta": zeta,
                "Cost": 0,
                "System Type": "Hybrid",
                "Total Energy (J)": total_energy_rr,
                "Runtime (s)": total_runtime_rr,
            }
        )


results_df = pd.DataFrame(results)

results_df["Total Energy (kWh)"] = results_df["Total Energy (J)"] / 3600000

sns.set(style="whitegrid", context="talk", font_scale=1.2, palette="colorblind")

# Plot for Total Energy
g_energy = sns.relplot(
    data=results_df,
    x="Zeta",
    y="Total Energy (kWh)",
    hue="Method",
    col="Dataset",
    col_wrap=3,
    kind="line",
    # marker="o",
    facet_kws={"sharey": False, "sharex": True},
)
g_energy.set_titles("{col_name}")
plt.savefig("heterogeneous-offline-energy.pdf", bbox_inches="tight")

# Plot for Total Energy
g_energy = sns.relplot(
    data=results_df,
    x="Zeta",
    y="Runtime (s)",
    hue="Method",
    col="Dataset",
    col_wrap=3,
    kind="line",
    # marker="o",
    facet_kws={"sharey": False, "sharex": True},
)
g_energy.set_titles("{col_name}")
plt.savefig("heterogeneous-offline-runtime.pdf", bbox_inches="tight")

# for ax in g_energy.axes.flat:
#     ax.set_xscale("log", base=2)
#     if ax.get_title() == "Alpaca":
#         max_alpaca_energy = results_df[
#             (results_df["Dataset"] == "Alpaca") & (results_df["Method"] == "AMD+A100")
#         ]["Total Energy (kWh)"].max()
#         min_alpaca_energy = results_df[
#             (results_df["Dataset"] == "Alpaca") & (results_df["Method"] == "M1-Pro")
#         ]["Total Energy (kWh)"].min()
#         ax.set_ylim(math.floor(min_alpaca_energy) - 1, math.ceil(max_alpaca_energy) + 1)
#         ax.set_yticks(
#             np.linspace(math.floor(min_alpaca_energy), math.ceil(max_alpaca_energy), 5)
#         )
#     elif ax.get_title() == "GSM8K":
#         max_gsm8k_energy = results_df[
#             (results_df["Dataset"] == "GSM8K") & (results_df["Method"] == "AMD+A100")
#         ]["Total Energy (kWh)"].max()
#         min_gsm8k_energy = results_df[
#             (results_df["Dataset"] == "GSM8K") & (results_df["Method"] == "M1-Pro")
#         ]["Total Energy (kWh)"].min()
#         ax.set_ylim(
#             math.floor(min_gsm8k_energy) - 0.2, math.ceil(max_gsm8k_energy) + 0.2
#         )
#         ax.set_yticks(
#             np.linspace(math.floor(min_gsm8k_energy), math.ceil(max_gsm8k_energy), 5)
#         )
#     elif ax.get_title() == "Python Codes 25K":
#         max_python_codes_energy = results_df[
#             (results_df["Dataset"] == "Python Codes 25K")
#             & (results_df["Method"] == "AMD+A100")
#         ]["Total Energy (kWh)"].max()
#         min_python_codes_energy = results_df[
#             (results_df["Dataset"] == "Python Codes 25K")
#             & (results_df["Method"] == "M1-Pro")
#         ]["Total Energy (kWh)"].min()
#         ax.set_ylim(
#             math.floor(min_python_codes_energy) - 0.5,
#             math.ceil(max_python_codes_energy) + 0.5,
#         )
#         ax.set_yticks(
#             np.linspace(
#                 math.floor(min_python_codes_energy),
#                 math.ceil(max_python_codes_energy),
#                 5,
#             )
#         )
# plt.savefig("heterogeneous-threshold-energy.pdf", bbox_inches="tight")

# # Plot for Runtime
# sns.set(style="whitegrid", context="talk", font_scale=1.2, palette="colorblind")
# g_runtime = sns.relplot(
#     data=results_df,
#     x="Threshold",
#     y="Runtime (s) x 1e5",
#     hue="Method",
#     style="System Type",
#     col="Dataset",
#     col_wrap=3,
#     kind="line",
#     # marker="o",
#     facet_kws={"sharey": False, "sharex": True},
# )
# g_runtime.set_titles("{col_name}")
# for ax in g_runtime.axes.flat:
#     ax.set_xscale("log", base=2)
#     if ax.get_title() == "Alpaca":
#         min_alpaca_runtime = results_df[
#             (results_df["Dataset"] == "Alpaca") & (results_df["Method"] == "AMD+A100")
#         ]["Runtime (s) x 1e5"].max()
#         max_alpaca_runtime = results_df[
#             (results_df["Dataset"] == "Alpaca") & (results_df["Method"] == "M1-Pro")
#         ]["Runtime (s) x 1e5"].min()
#         ax.set_ylim(
#             math.floor(min_alpaca_runtime) - 1, math.ceil(max_alpaca_runtime) + 1
#         )
#         ax.set_yticks(
#             np.linspace(
#                 math.floor(min_alpaca_runtime), math.ceil(max_alpaca_runtime), 5
#             )
#         )
#     elif ax.get_title() == "GSM8K":
#         min_gsm8k_runtime = results_df[
#             (results_df["Dataset"] == "GSM8K") & (results_df["Method"] == "AMD+A100")
#         ]["Runtime (s) x 1e5"].max()
#         max_gsm8k_runtime = results_df[
#             (results_df["Dataset"] == "GSM8K") & (results_df["Method"] == "M1-Pro")
#         ]["Runtime (s) x 1e5"].min()
#         ax.set_ylim(math.floor(min_gsm8k_runtime), math.ceil(max_gsm8k_runtime))
#         ax.set_yticks(
#             np.linspace(math.floor(min_gsm8k_runtime), math.ceil(max_gsm8k_runtime), 5)
#         )
#     elif ax.get_title() == "Python Codes 25K":
#         min_python_codes_runtime = results_df[
#             (results_df["Dataset"] == "Python Codes 25K")
#             & (results_df["Method"] == "AMD+A100")
#         ]["Runtime (s) x 1e5"].max()
#         max_python_codes_runtime = results_df[
#             (results_df["Dataset"] == "Python Codes 25K")
#             & (results_df["Method"] == "M1-Pro")
#         ]["Runtime (s) x 1e5"].min()
#         ax.set_ylim(
#             math.floor(min_python_codes_runtime) - 1,
#             math.ceil(max_python_codes_runtime) + 1,
#         )
#         ax.set_yticks(
#             np.linspace(
#                 math.floor(min_python_codes_runtime),
#                 math.ceil(max_python_codes_runtime),
#                 5,
#             )
#         )
# plt.savefig("heterogeneous-threshold-runtime.pdf", bbox_inches="tight")


# # Plot for Total Energy
# g_energy = sns.relplot(
#     data=results_df,
#     x="Threshold",
#     y="Total Energy (kWh)",
#     hue="Method",
#     style="System Type",
#     col="Dataset",
#     col_wrap=3,
#     kind="line",
#     # marker="o",
#     facet_kws={"sharey": False, "sharex": True},
# )
# g_energy.set_titles("{col_name}")
# for ax in g_energy.axes.flat:
#     if ax.get_title() == "Alpaca":
#         max_alpaca_energy = results_df[
#             (results_df["Dataset"] == "Alpaca") & (results_df["Method"] == "AMD+A100")
#         ]["Total Energy (kWh)"].max()
#         min_alpaca_energy = results_df[
#             (results_df["Dataset"] == "Alpaca") & (results_df["Method"] == "M1-Pro")
#         ]["Total Energy (kWh)"].min()
#         ax.set_ylim(math.floor(min_alpaca_energy) - 1, math.ceil(max_alpaca_energy) + 1)
#         ax.set_yticks(
#             np.linspace(math.floor(min_alpaca_energy), math.ceil(max_alpaca_energy), 5)
#         )
#     elif ax.get_title() == "GSM8K":
#         max_gsm8k_energy = results_df[
#             (results_df["Dataset"] == "GSM8K") & (results_df["Method"] == "AMD+A100")
#         ]["Total Energy (kWh)"].max()
#         min_gsm8k_energy = results_df[
#             (results_df["Dataset"] == "GSM8K") & (results_df["Method"] == "M1-Pro")
#         ]["Total Energy (kWh)"].min()
#         ax.set_ylim(
#             math.floor(min_gsm8k_energy) - 0.2, math.ceil(max_gsm8k_energy) + 0.2
#         )
#         ax.set_yticks(
#             np.linspace(math.floor(min_gsm8k_energy), math.ceil(max_gsm8k_energy), 5)
#         )
#     elif ax.get_title() == "Python Codes 25K":
#         max_python_codes_energy = results_df[
#             (results_df["Dataset"] == "Python Codes 25K")
#             & (results_df["Method"] == "AMD+A100")
#         ]["Total Energy (kWh)"].max()
#         min_python_codes_energy = results_df[
#             (results_df["Dataset"] == "Python Codes 25K")
#             & (results_df["Method"] == "M1-Pro")
#         ]["Total Energy (kWh)"].min()
#         ax.set_ylim(
#             math.floor(min_python_codes_energy) - 0.5,
#             math.ceil(max_python_codes_energy) + 0.5,
#         )
#         ax.set_yticks(
#             np.linspace(
#                 math.floor(min_python_codes_energy),
#                 math.ceil(max_python_codes_energy),
#                 5,
#             )
#         )
# plt.savefig("heterogeneous-threshold-energy-linear.pdf", bbox_inches="tight")

# # Plot for Runtime
# sns.set(style="whitegrid", context="talk", font_scale=1.2, palette="colorblind")
# g_runtime = sns.relplot(
#     data=results_df,
#     x="Threshold",
#     y="Runtime (s) x 1e5",
#     hue="Method",
#     style="System Type",
#     col="Dataset",
#     col_wrap=3,
#     kind="line",
#     # marker="o",
#     facet_kws={"sharey": False, "sharex": True},
# )
# g_runtime.set_titles("{col_name}")
# for ax in g_runtime.axes.flat:
#     if ax.get_title() == "Alpaca":
#         min_alpaca_runtime = results_df[
#             (results_df["Dataset"] == "Alpaca") & (results_df["Method"] == "AMD+A100")
#         ]["Runtime (s) x 1e5"].max()
#         max_alpaca_runtime = results_df[
#             (results_df["Dataset"] == "Alpaca") & (results_df["Method"] == "M1-Pro")
#         ]["Runtime (s) x 1e5"].min()
#         ax.set_ylim(
#             math.floor(min_alpaca_runtime) - 1, math.ceil(max_alpaca_runtime) + 1
#         )
#         ax.set_yticks(
#             np.linspace(
#                 math.floor(min_alpaca_runtime), math.ceil(max_alpaca_runtime), 5
#             )
#         )
#     elif ax.get_title() == "GSM8K":
#         min_gsm8k_runtime = results_df[
#             (results_df["Dataset"] == "GSM8K") & (results_df["Method"] == "AMD+A100")
#         ]["Runtime (s) x 1e5"].max()
#         max_gsm8k_runtime = results_df[
#             (results_df["Dataset"] == "GSM8K") & (results_df["Method"] == "M1-Pro")
#         ]["Runtime (s) x 1e5"].min()
#         ax.set_ylim(math.floor(min_gsm8k_runtime), math.ceil(max_gsm8k_runtime))
#         ax.set_yticks(
#             np.linspace(math.floor(min_gsm8k_runtime), math.ceil(max_gsm8k_runtime), 5)
#         )
#     elif ax.get_title() == "Python Codes 25K":
#         min_python_codes_runtime = results_df[
#             (results_df["Dataset"] == "Python Codes 25K")
#             & (results_df["Method"] == "AMD+A100")
#         ]["Runtime (s) x 1e5"].max()
#         max_python_codes_runtime = results_df[
#             (results_df["Dataset"] == "Python Codes 25K")
#             & (results_df["Method"] == "M1-Pro")
#         ]["Runtime (s) x 1e5"].min()
#         ax.set_ylim(
#             math.floor(min_python_codes_runtime) - 1,
#             math.ceil(max_python_codes_runtime) + 1,
#         )
#         ax.set_yticks(
#             np.linspace(
#                 math.floor(min_python_codes_runtime),
#                 math.ceil(max_python_codes_runtime),
#                 5,
#             )
#         )
# plt.savefig("heterogeneous-threshold-runtime-linear.pdf", bbox_inches="tight")
