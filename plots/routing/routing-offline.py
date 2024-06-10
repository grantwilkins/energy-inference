import pulp as lp
from pulp import GUROBI
from datasets import load_dataset
from transformers import AutoTokenizer
import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


# Define the problem
problem = lp.LpProblem("LLM_Scheduling_Optimization", lp.LpMinimize)

dataset = load_dataset("vicgalle/alpaca-gpt4")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
lengths_instructions = [
    len(tokenizer.encode(x)) for x in dataset["train"]["instruction"]
]
lengths_inputs = [len(tokenizer.encode(x)) for x in dataset["train"]["input"]]
lengths_outputs = [len(tokenizer.encode(x)) for x in dataset["train"]["output"]]
lengths = [x + y for x, y in zip(lengths_instructions, lengths_inputs)]

Q = list(zip(lengths, lengths_outputs))

Q = Q[:200]
total_tokens = sum(t_in + t_out for t_in, t_out in Q)
# Number of queries
num_queries = len(Q)

N_system = 16000

zeta_values = np.arange(0, 1.1, 0.1)
results = []


def normalize(value, min_value, max_value):
    return (
        (value - min_value) / (max_value - min_value) if max_value != min_value else 0
    )


models = [
    "Llama-2 (7B)",
    "Llama-2 (13B)",
    "Llama-2 (70B)",
]
# Number of models
K = len(models)  # total number of models
# System total GPUs
# GPUs required for each LLM instance
n_K = {
    "Llama-2 (7B)": 1,
    "Llama-2 (13B)": 1,
    "Llama-2 (70B)": 4,
}  # Dictionary with model index as keys and GPU requirements as values
# Proportion of system assigned to each model
gamma_K = {
    "Llama-2 (7B)": 0.05,
    "Llama-2 (13B)": 0.2,
    "Llama-2 (70B)": 0.75,
}  # Dictionary with model index as keys and gamma values as values

# assert sum(gamma_K.values()) == 1.0

accuracies = {"Llama-2 (7B)": 0.5097, "Llama-2 (13B)": 0.5569, "Llama-2 (70B)": 0.6452}

# Coefficients for energy models
alpha_coeffs = {
    "Llama-2 (7B)": {
        "alpha_0": -3.894991,
        "alpha_1": 31.522735,
        "alpha_2": 0.042712,
    },
    "Llama-2 (13B)": {
        "alpha_0": -6.794323,
        "alpha_1": 56.008311,
        "alpha_2": 0.072861,
    },
    "Llama-2 (70B)": {
        "alpha_0": -12.0294,
        "alpha_1": 414.8197,
        "alpha_2": 0.3145,
    },
    # Add other models as needed
}

# Coefficients for runtime models
beta_coeffs = {
    "Llama-2 (7B)": {
        "beta_0": -0.010021,
        "beta_1": 0.083515,
        "beta_2": 0.000107,
    },
    "Llama-2 (13B)": {
        "beta_0": -0.017357,
        "beta_1": 0.142605,
        "beta_2": 0.000185,
    },
    "Llama-2 (70B)": {
        "beta_0": -0.031204,
        "beta_1": 0.703173,
        "beta_2": 0.000533,
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


max_values = {}
for model in models:
    max_runtime = 0
    max_energy = 0
    max_accuracy = 0
    for t_in, t_out in Q:
        current_runtime = runtime_cost(model, t_in, t_out)
        current_energy = energy_cost(model, t_in, t_out)
        current_accuracy = (t_in + t_out) * accuracies[model]

        if current_runtime > max_runtime:
            max_runtime = current_runtime
        if current_energy > max_energy:
            max_energy = current_energy
        if current_accuracy > max_accuracy:
            max_accuracy = current_accuracy

    max_values[model] = {
        "max_runtime": max_runtime,
        "max_energy": max_energy,
        "max_accuracy": max_accuracy,
    }
print(max_values)

min_values = {}
for model in models:
    min_values[model] = {
        "min_runtime": 0,
        "min_energy": 0,
        "min_accuracy": 0,
    }


def normalized_energy_cost(K, t_in, t_out):
    energy_min = min_values[K]["min_energy"]
    energy_max = max_values[K]["max_energy"]
    return normalize(energy_cost(K, t_in, t_out), energy_min, energy_max)


def normalized_runtime_cost(K, t_in, t_out):
    runtime_min = min_values[K]["min_runtime"]
    runtime_max = max_values[K]["max_runtime"]
    return normalize(runtime_cost(K, t_in, t_out), runtime_min, runtime_max)


def normalized_accuracy(K, t_in, t_out):
    accuracy_min = min_values[K]["min_accuracy"]
    accuracy_max = max_values[K]["max_accuracy"]
    return normalize((t_in + t_out) * accuracies[K], accuracy_min, accuracy_max)


# Variables for each query assigned to each model, binary
x = {
    (k, i): lp.LpVariable(f"x_{k}_{i}", cat="Binary")
    for k in models
    for i in range(num_queries)
}


# Define accuracy, energy, and runtime functions (assuming these functions are defined or approximated)
def a_K(k, queries):
    return sum(accuracies[k] * (t_in + t_out) for t_in, t_out in queries)


def e_K(k, queries):
    return sum(energy_cost(k, t_in, t_out) for t_in, t_out in queries)


def r_K(k, queries):
    return sum(runtime_cost(k, t_in, t_out) for t_in, t_out in queries)


# Round Robin Routing
round_robin_assignments = {}
for i in range(num_queries):
    model_index = i % K
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


# Function to calculate total energy, runtime, and accuracy
def calculate_metrics(assignments):
    total_energy = 0
    total_runtime = 0
    total_accuracy = 0
    for model, queries in assignments.items():
        for t_in, t_out in queries:
            total_energy += energy_cost(model, t_in, t_out)
            total_runtime += runtime_cost(model, t_in, t_out)
            total_accuracy += (t_in + t_out) * accuracies[model]
    return total_energy, total_runtime, total_accuracy


# Calculate metrics for Round Robin
rr_energy, rr_runtime, rr_accuracy = calculate_metrics(round_robin_assignments)
rr_accuracy = (rr_accuracy / sum(t_in + t_out for t_in, t_out in Q)) * 100
print("Round Robin Routing:")
print("Total Energy:", rr_energy)
print("Total Runtime:", rr_runtime)
print("Total Accuracy:", rr_accuracy)
for zeta in zeta_values:
    results.append(
        {
            "Zeta": zeta,
            "Total Energy (J)": rr_energy,
            "Accuracy (%)": rr_accuracy,
            "Runtime (s)": rr_runtime,
            "Mean Runtime (s)": rr_runtime / num_queries,
            "Assigned to Llama-2 (7B)": len(
                round_robin_assignments.get("Llama-2 (7B)", 0)
            ),
            "Assigned to Llama-2 (13B)": len(
                round_robin_assignments.get("Llama-2 (13B)", 0)
            ),
            "Assigned to Llama-2 (70B)": len(
                round_robin_assignments.get("Llama-2 (70B)", 0)
            ),
            "Method": "Round Robin",
            "LLMs": "Ensemble",
            "Scheduler": "True",
        }
    )

# Calculate metrics for Random Routing
rand_energy, rand_runtime, rand_accuracy = calculate_metrics(random_assignments)
rand_accuracy = (rand_accuracy / sum(t_in + t_out for t_in, t_out in Q)) * 100
print("Random Routing:")
print("Total Energy:", rand_energy)
print("Total Runtime:", rand_runtime)
print("Total Accuracy:", rand_accuracy)
for zeta in zeta_values:
    results.append(
        {
            "Zeta": zeta,
            "Total Energy (J)": rand_energy,
            "Accuracy (%)": rand_accuracy,
            "Runtime (s)": rand_runtime,
            "Mean Runtime (s)": rand_runtime / num_queries,
            "Assigned to Llama-2 (7B)": len(random_assignments.get("Llama-2 (7B)", 0)),
            "Assigned to Llama-2 (13B)": len(
                random_assignments.get("Llama-2 (13B)", 0)
            ),
            "Assigned to Llama-2 (70B)": len(
                random_assignments.get("Llama-2 (70B)", 0)
            ),
            "Method": "Random",
            "LLMs": "Ensemble",
            "Scheduler": "True",
        }
    )

# Calculate and plot the energy/accuracy for using each model only
for model in models:
    # Assume all queries are assigned to this model
    all_queries_assigned = [(Q[i][0], Q[i][1]) for i in range(num_queries)]
    model_runtime = r_K(model, all_queries_assigned)
    model_energy = e_K(model, all_queries_assigned)
    model_accuracy = a_K(model, all_queries_assigned)
    model_tokens = sum(t_in + t_out for t_in, t_out in Q)
    accuracy = (model_accuracy / model_tokens) * 100
    print(f"Model {model}: Total Energy {model_energy:.2f}")
    print(f"Model {model}: Total Accuracy {accuracies[model]:.2f}")
    print(f"Model {model}: Total Runtime {model_runtime:.2f}")
    for zeta in zeta_values:
        results.append(
            {
                "Zeta": zeta,
                "Total Energy (J)": model_energy,
                "Accuracy (%)": accuracy,
                "Runtime (s)": model_runtime,
                "Mean Runtime (s)": model_runtime / num_queries,
                "Assigned to Llama-2 (7B)": (
                    num_queries if model == "Llama-2 (7B)" else 0
                ),
                "Assigned to Llama-2 (13B)": (
                    num_queries if model == "Llama-2 (13B)" else 0
                ),
                "Assigned to Llama-2 (70B)": (
                    num_queries if model == "Llama-2 (70B)" else 0
                ),
                "Method": model,
                "LLMs": "Single",
                "Scheduler": "False",
            }
        )


def dynamic_normalized_cost(cost_func, K, t_in, t_out, current_values):
    min_val = min(current_values) if current_values else 0
    max_val = max(current_values) if current_values else 1
    return normalize(cost_func(K, t_in, t_out), min_val, max_val)


for zeta in zeta_values:
    current_energy_costs = [
        energy_cost(k, t_in, t_out) for k in models for t_in, t_out in Q
    ]
    current_runtime_costs = [
        runtime_cost(k, t_in, t_out) for k in models for t_in, t_out in Q
    ]
    current_accuracies = [
        (t_in + t_out) * accuracies[k] for k in models for t_in, t_out in Q
    ]

    problem += (
        lp.lpSum(
            [
                zeta
                * (
                    dynamic_normalized_cost(
                        energy_cost, k, t_in, t_out, current_energy_costs
                    )
                )
                * x[(k, i)]
                - (1 - zeta)
                * dynamic_normalized_cost(
                    lambda K, t_in, t_out: (t_in + t_out) * accuracies[K],
                    k,
                    t_in,
                    t_out,
                    current_accuracies,
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

    status = problem.solve()
    print("Status:", lp.LpStatus[status])

    if status == lp.LpStatusInfeasible:
        problem.writeLP("model_debug.lp")
        print("Check the generated 'model_debug.lp' file for more details.")

    total_accuracy = 0
    num_assigned = {}
    for k in models:
        assigned_queries = [
            (Q[i][0], Q[i][1]) for i in range(num_queries) if x[(k, i)].varValue == 1
        ]
        total_accuracy += a_K(k, assigned_queries)
        num_assigned[k] = len(assigned_queries)
        print(f"Model {k}: Queries {len(assigned_queries)}")

    total_energy = 0
    total_runtime = 0
    for k in models:
        assigned_queries = [
            (Q[i][0], Q[i][1]) for i in range(num_queries) if x[(k, i)].varValue == 1
        ]
        total_energy += e_K(k, assigned_queries)
        total_runtime += r_K(k, assigned_queries)
        print(f"Model {k}: Total Energy {total_energy:.2f}")

    print(f"Total Energy: {total_energy:.2f}")
    print(f"Total Accuracy: {total_accuracy/total_tokens*100:.2f}")
    results.append(
        {
            "Zeta": zeta,
            "Total Energy (J)": total_energy,
            "Accuracy (%)": total_accuracy / total_tokens * 100,
            "Runtime (s)": total_runtime,
            "Mean Runtime (s)": total_runtime / num_queries,
            "Assigned to Llama-2 (7B)": num_assigned.get("Llama-2 (7B)", 0),
            "Assigned to Llama-2 (13B)": num_assigned.get("Llama-2 (13B)", 0),
            "Assigned to Llama-2 (70B)": num_assigned.get("Llama-2 (70B)", 0),
            "Method": "Offline",
            "LLMs": "Ensemble",
            "Scheduler": "True",
        }
    )

    # Clear the problem
    problem = lp.LpProblem("LLM_Scheduling_Optimization", lp.LpMinimize)

df = pd.DataFrame(results)
df["Total Energy (kWh)"] = df["Total Energy (J)"] / 3.6e6

plt.figure(figsize=(6, 6))
sns.set(style="whitegrid", context="talk", font_scale=1.5)
sns.set_palette("colorblind")
sns.lineplot(
    data=df,
    x="Zeta",
    y="Total Energy (kWh)",
    hue="Method",
    legend=False,
    linewidth=3,
    hue_order=[
        "Llama-2 (7B)",
        "Llama-2 (13B)",
        "Llama-2 (70B)",
        "Round Robin",
        "Random",
        "Offline",
    ],
    style="Scheduler",
    style_order=["True", "False"],
)
# plt.legend(bbox_to_anchor=(1, 0.5), loc="center left", frameon=False)
plt.xlim(0, 1)
plt.ylim(0, 4)
# plt.yscale("log")
plt.savefig(
    f"routing-offline-energy-{gamma_K['Llama-2 (7B)']}-{gamma_K['Llama-2 (13B)']}-{gamma_K['Llama-2 (70B)']}.pdf",
    bbox_inches="tight",
)

plt.figure(figsize=(6, 6))
sns.set(style="whitegrid", context="talk", font_scale=1.5)
sns.set_palette("colorblind")
sns.lineplot(
    data=df,
    x="Zeta",
    y="Accuracy (%)",
    hue="Method",
    linewidth=3,
    hue_order=[
        "Llama-2 (7B)",
        "Llama-2 (13B)",
        "Llama-2 (70B)",
        "Round Robin",
        "Random",
        "Offline",
    ],
    style="Scheduler",
    style_order=["True", "False"],
)
plt.xlim(0, 1)
plt.ylim(50, 65)
plt.legend(bbox_to_anchor=(1, 0.5), loc="center left", frameon=False)
plt.savefig(
    f"routing-offline-accuracy-{gamma_K['Llama-2 (7B)']}-{gamma_K['Llama-2 (13B)']}-{gamma_K['Llama-2 (70B)']}.pdf",
    bbox_inches="tight",
)

plt.figure(figsize=(6, 6))
sns.set(style="whitegrid", context="talk", font_scale=1.5)
sns.set_palette("colorblind")
sns.lineplot(
    data=df,
    x="Zeta",
    y="Mean Runtime (s)",
    hue="Method",
    legend=False,
    linewidth=3,
    hue_order=[
        "Llama-2 (7B)",
        "Llama-2 (13B)",
        "Llama-2 (70B)",
        "Round Robin",
        "Random",
        "Offline",
    ],
    style="Scheduler",
    style_order=["True", "False"],
)
plt.xlim(0, 1)
plt.ylim(0, 100)
# plt.legend(bbox_to_anchor=(1, 0.5), loc="center left", frameon=False)
plt.savefig(
    f"routing-offline-runtime-{gamma_K['Llama-2 (7B)']}-{gamma_K['Llama-2 (13B)']}-{gamma_K['Llama-2 (70B)']}.pdf",
    bbox_inches="tight",
)
