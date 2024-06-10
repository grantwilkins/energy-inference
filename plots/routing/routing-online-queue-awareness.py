import pulp as lp
import numpy as np
import random
from collections import deque
from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from random import shuffle


def normalize(value, min_value, max_value):
    return (
        (value - min_value) / (max_value - min_value) if max_value != min_value else 0
    )


# Dictionary of energy calculation functions


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


def normalized_energy_cost(K, t_in, t_out):
    energy_min = (
        min(historical_data[K]["Energy"])
        if historical_data[K]["Energy"]
        else min_values[K]["min_energy"]
    )
    energy_max = (
        max(historical_data[K]["Energy"])
        if historical_data[K]["Energy"]
        else max_values[K]["max_energy"]
    )
    return normalize(energy_cost(K, t_in, t_out), energy_min, energy_max)


def normalized_runtime_cost(K, t_in, t_out):
    runtime_min = min_values[K]["min_runtime"]
    runtime_max = max_values[K]["max_runtime"]
    return normalize(runtime_cost(K, t_in, t_out), runtime_min, runtime_max)


def normalized_accuracy(K, t_in, t_out):
    accuracy_min = (
        min(historical_data[K]["Accuracy"])
        if historical_data[K]["Accuracy"]
        else min_values[K]["min_accuracy"]
    )
    accuracy_max = (
        max(historical_data[K]["Accuracy"])
        if historical_data[K]["Accuracy"]
        else max_values[K]["max_accuracy"]
    )
    return normalize((t_in + t_out) * accuracies[K], accuracy_min, accuracy_max)


def accuracy_cost(K, t_in, t_out):
    return accuracies[K] * (t_in + t_out)


def calculate_cost(K, t_in, t_out, zeta):
    return zeta * (normalized_energy_cost(K, t_in, t_out)) - (
        1 - zeta
    ) * normalized_accuracy(K, t_in, t_out)


# Function to update system state and process queries in queues
def update_system_state(t, zeta):
    for K in models:
        # Process queries in GPU queue if their projected end time has been reached
        while gpu_queues[K] and gpu_queues[K][0][1] <= t:
            query, end_time = gpu_queues[K].popleft()
            t_in, t_out = query
            current_cost[K] -= calculate_cost(K, t_in, t_out, zeta)
            # print(f"Completed query {query} from {K} at time {t}")
            if queues[K]:
                if len(gpu_queues[K]) < gpu_capacities[K]:
                    query = queues[K].popleft()
                    wait_times[K].append(t - query[1])
                    gpu_queues[K].append(
                        (query[0], t + runtime_cost(K, query[0][0], query[0][1]))
                    )


def handle_query(query, t, zeta):
    min_cost = float("inf")
    selected_model = None
    t_in, t_out, arrival_time = query
    for K in models:
        penalty = len(queues[K]) * 0.1
        cost_with_query = (
            current_cost[K] + calculate_cost(K, t_in, t_out, zeta) + penalty
        )
        if cost_with_query < min_cost:
            min_cost = cost_with_query
            selected_model = K
    if len(gpu_queues[selected_model]) < gpu_capacities[selected_model]:
        projected_end_time = t + runtime_cost(selected_model, t_in, t_out)
        gpu_queues[selected_model].append((query[:2], projected_end_time))
        wait_times[selected_model].append(0)
    else:
        queues[selected_model].append((query[:2], t))
    queue_history[selected_model].append((query[:2], t))
    current_cost[selected_model] += calculate_cost(selected_model, t_in, t_out, zeta)
    if len(historical_data[selected_model]["Energy"]) % 10 == 0:
        for K in models:
            current_cost[K] = 0
    historical_data[selected_model]["Energy"].append(
        energy_cost(selected_model, t_in, t_out)
    )
    historical_data[selected_model]["Runtime"].append(
        runtime_cost(selected_model, t_in, t_out)
    )
    historical_data[selected_model]["Accuracy"].append(
        accuracy_cost(selected_model, t_in, t_out)
    )
    if len(historical_data[selected_model]["Energy"]) > 100:
        historical_data[selected_model]["Energy"].popleft()
        historical_data[selected_model]["Runtime"].popleft()
        historical_data[selected_model]["Accuracy"].popleft()


def handle_query_random(query, t):
    t_in, t_out, arrival_time = query
    selected_model = random.choice(models)  # Select a model at random
    if len(gpu_queues[selected_model]) < gpu_capacities[selected_model]:
        projected_end_time = t + runtime_cost(selected_model, t_in, t_out)
        gpu_queues[selected_model].append((query[:2], projected_end_time))
        wait_times[selected_model].append(0)
    else:
        queues[selected_model].append((query[:2], t))
    queue_history[selected_model].append((query[:2], t))


def handle_query_round_robin(query, t, current_index):
    t_in, t_out, arrival_time = query
    selected_model = models[current_index]
    if len(gpu_queues[selected_model]) < gpu_capacities[selected_model]:
        projected_end_time = t + runtime_cost(selected_model, t_in, t_out)
        gpu_queues[selected_model].append((query[:2], projected_end_time))
        wait_times[selected_model].append(0)
    else:
        queues[selected_model].append((query[:2], t))
    queue_history[selected_model].append((query[:2], t))
    current_index = (current_index + 1) % len(models)
    return current_index


def simulate_queries_oursol(queries, zeta):
    for query in queries:
        t = query[2]
        handle_query(query, t, zeta)
        update_system_state(t, zeta)
        # print(f"System state updated at time {t}")


def simultate_queries_random(queries):
    for query in queries:
        t = query[2]
        handle_query_random(query, t)
        update_system_state(t, 0)


def simulate_queries_round_robin(queries):
    current_index = 0
    for query in queries:
        t = query[2]
        current_index = handle_query_round_robin(query, t, current_index)
        update_system_state(t, 0)


def analyze_results():
    total_energy = sum(
        energy_cost(K, t_in, t_out)
        for K in models
        for t_in, t_out in (q[0] for q in queue_history[K])
    )
    total_runtime = sum(
        runtime_cost(K, t_in, t_out)
        for K in models
        for t_in, t_out in (q[0] for q in queue_history[K])
    )
    total_accuracy = sum(
        accuracy_cost(K, t_in, t_out)
        for K in models
        for t_in, t_out in (q[0] for q in queue_history[K])
    )

    total_num_tokens = sum(
        t_in + t_out for K in models for t_in, t_out in (q[0] for q in queue_history[K])
    )
    total_accuracy /= total_num_tokens

    num_assigned_to_Llama_2_7B = len(queue_history["Llama-2 (7B)"])
    num_assigned_to_Llama_2_13B = len(queue_history["Llama-2 (13B)"])
    num_assigned_to_Llama_2_70B = len(queue_history["Llama-2 (70B)"])

    print(f"Total Energy: {total_energy:.2f}")
    print(f"Total Runtime: {total_runtime:.2f}")
    print(f"Total Accuracy: {total_accuracy:.2f}")
    print(f"Number Assigned To Llama-2 (7B): {len(queue_history['Llama-2 (7B)'])}")
    print(f"Number Assigned To Llama-2 (13B): {len(queue_history['Llama-2 (13B)'])}")
    print(f"Number Assigned To Llama-2 (70B): {len(queue_history['Llama-2 (70B)'])}")

    return (
        total_energy,
        total_runtime,
        total_accuracy,
        num_assigned_to_Llama_2_7B,
        num_assigned_to_Llama_2_13B,
        num_assigned_to_Llama_2_70B,
    )


def main():
    global queues, current_cost, queue_history, historical_data, gpu_queues, gpu_capacities, arrival_times, Q, models, accuracies, alpha_coeffs, beta_coeffs, max_values, min_values, wait_times
    num_gpus = 16000
    # num_queries = 10000
    # Define models and parameters
    models = ["Llama-2 (7B)", "Llama-2 (13B)", "Llama-2 (70B)"]
    K = len(models)  # total number of models
    n_K = {"Llama-2 (7B)": 1, "Llama-2 (13B)": 1, "Llama-2 (70B)": 4}
    gamma_K = {
        "Llama-2 (7B)": 0.05,
        "Llama-2 (13B)": 0.2,
        "Llama-2 (70B)": 0.75,
    }
    historical_data = {
        K: {"Energy": deque(), "Runtime": deque(), "Accuracy": deque()} for K in models
    }
    # assert sum(gamma_K.values()) == 1.0

    accuracies = {
        "Llama-2 (7B)": 0.5097,
        "Llama-2 (13B)": 0.5569,
        "Llama-2 (70B)": 0.6452,
    }

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

    max_values = {}

    dataset = load_dataset("vicgalle/alpaca-gpt4")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    lengths_instructions = [
        len(tokenizer.encode(x)) for x in dataset["train"]["instruction"]
    ]
    lengths_inputs = [len(tokenizer.encode(x)) for x in dataset["train"]["input"]]
    lengths_outputs = [len(tokenizer.encode(x)) for x in dataset["train"]["output"]]
    lengths = [x + y for x, y in zip(lengths_instructions, lengths_inputs)]
    num_queries = len(lengths)

    arrival_times = sorted(
        np.random.uniform(0, 1000, num_queries)
    )  # Simulate arrivals over time
    Q = list(zip(lengths, lengths_outputs, arrival_times))
    queues = {K: deque() for K in models}
    gpu_queues = {K: deque() for K in models}
    queue_history = {K: [] for K in models}
    current_cost = {K: 0 for K in models}
    wait_times = {K: [] for K in models}
    gpu_capacities = {K: int(gamma_K[K] * num_gpus) for K in models}
    max_input_tokens = 2048
    max_output_tokens = 2048

    for model in models:
        max_runtime = runtime_cost(model, max_input_tokens, max_output_tokens)
        max_energy = energy_cost(model, max_input_tokens, max_output_tokens)
        max_accuracy = accuracy_cost(model, max_input_tokens, max_output_tokens)
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

    data = []

    zeta_values = np.arange(0, 1.1, 0.1)
    for zeta in zeta_values:
        simulate_queries_oursol(Q, zeta)
        print(f"Results for Zeta = {zeta}")
        energy, runtime, accuracy, num7b, num13b, num70b = analyze_results()
        average_wait_time = 0
        for K in models:
            average_wait_time += sum(wait_times[K])
        average_wait_time /= sum(len(wait_times[K]) for K in models)
        data.append(
            {
                "Zeta": zeta,
                "Total Energy (kWh)": energy / 3.6e6,
                "Accuracy (%)": accuracy * 100,
                "Runtime (s)": runtime,
                "Num Queries Llama-2 (7B)": num7b,
                "Num Queries Llama-2 (13B)": num13b,
                "Num Queries Llama-2 (70B)": num70b,
                "Average Wait Time (s)": average_wait_time,
                "Method": "Online",
            }
        )
        queue_history = {K: [] for K in models}
        current_cost = {K: 0 for K in models}
        queues = {K: deque() for K in models}
        gpu_queues = {K: deque() for K in models}
        wait_times = {K: [] for K in models}

    # Round-robin simulation
    assignment_history = {K: [] for K in models}
    queue_history = {K: [] for K in models}
    current_cost = {K: 0 for K in models}
    queues = {K: deque() for K in models}
    gpu_queues = {K: deque() for K in models}
    simulate_queries_round_robin(Q)
    energy, runtime, accuracy, num7b, num13b, num70b = analyze_results()
    average_wait_time = 0
    for K in models:
        average_wait_time += sum(wait_times[K])
    average_wait_time /= sum(len(wait_times[K]) for K in models)
    for zeta in zeta_values:
        data.append(
            {
                "Zeta": zeta,
                "Total Energy (kWh)": energy / 3.6e6,
                "Accuracy (%)": accuracy * 100,
                "Runtime (s)": runtime,
                "Num Queries Llama-2 (7B)": num7b,
                "Num Queries Llama-2 (13B)": num13b,
                "Num Queries Llama-2 (70B)": num70b,
                "Average Wait Time (s)": average_wait_time,
                "Method": "Round Robin",
            }
        )

    # Random simulation
    assignment_history = {K: [] for K in models}
    queue_history = {K: [] for K in models}
    current_cost = {K: 0 for K in models}
    queues = {K: deque() for K in models}
    gpu_queues = {K: deque() for K in models}
    wait_times = {K: [] for K in models}
    simultate_queries_random(Q)
    energy, runtime, accuracy, num7b, num13b, num70b = analyze_results()
    average_wait_time = 0
    for K in models:
        average_wait_time += sum(wait_times[K])
    average_wait_time /= sum(len(wait_times[K]) for K in models)
    for zeta in zeta_values:
        data.append(
            {
                "Zeta": zeta,
                "Total Energy (kWh)": energy / 3.6e6,
                "Accuracy (%)": accuracy * 100,
                "Runtime (s)": runtime,
                "Num Queries Llama-2 (7B)": num7b,
                "Num Queries Llama-2 (13B)": num13b,
                "Num Queries Llama-2 (70B)": num70b,
                "Average Wait Time (s)": average_wait_time,
                "Method": "Random",
            }
        )

    # Use one model for all queries
    for K in models:
        energy = sum(energy_cost(K, t_in, t_out) for t_in, t_out, _ in Q)
        runtime = sum(runtime_cost(K, t_in, t_out) for t_in, t_out, _ in Q)
        for zeta in zeta_values:
            data.append(
                {
                    "Zeta": zeta,
                    "Total Energy (kWh)": energy / 3.6e6,
                    "Accuracy (%)": accuracies[K] * 100,
                    "Runtime (s)": runtime,
                    "Num Queries Llama-2 (7B)": len(Q) if K == "Llama-2 (7B)" else 0,
                    "Num Queries Llama-2 (13B)": len(Q) if K == "Llama-2 (13B)" else 0,
                    "Num Queries Llama-2 (70B)": len(Q) if K == "Llama-2 (70B)" else 0,
                    "Method": K,
                }
            )

    df = pd.DataFrame(data)
    df.to_csv("simulation_results.csv", index=False)

    plt.figure(figsize=(6, 5))
    sns.set(style="whitegrid", context="talk", font_scale=1.2)
    sns.set_palette("colorblind")
    sns.lineplot(
        x="Zeta",
        y="Total Energy (kWh)",
        data=df,
        hue="Method",
        hue_order=[
            "Llama-2 (7B)",
            "Llama-2 (13B)",
            "Llama-2 (70B)",
            "Round Robin",
            "Random",
            "Online",
        ],
        legend=False,
    )
    plt.xlabel("Zeta")
    plt.xlim(0, 1)
    plt.ylabel("Total Energy (kWh)")
    plt.savefig(
        f"routing-online-awareness-energy-{gamma_K['Llama-2 (7B)']}-{gamma_K['Llama-2 (13B)']}-{gamma_K['Llama-2 (70B)']}.pdf",
        bbox_inches="tight",
    )

    plt.figure(figsize=(6, 5))
    sns.set(style="whitegrid", context="talk", font_scale=1.2)
    sns.set_palette("colorblind")
    sns.lineplot(
        x="Zeta",
        y="Accuracy (%)",
        data=df,
        hue="Method",
        hue_order=[
            "Llama-2 (7B)",
            "Llama-2 (13B)",
            "Llama-2 (70B)",
            "Round Robin",
            "Random",
            "Online",
        ],
    )
    plt.xlabel("Zeta")
    plt.xlim(0, 1)
    plt.legend(
        bbox_to_anchor=(1, 0.5),
        loc="center left",
        frameon=False,
    )
    plt.savefig(
        f"routing-online-awareness-accuracy-{gamma_K['Llama-2 (7B)']}-{gamma_K['Llama-2 (13B)']}-{gamma_K['Llama-2 (70B)']}.pdf",
        bbox_inches="tight",
    )

    plt.figure(figsize=(6, 5))
    sns.set(style="whitegrid", context="talk", font_scale=1.2)
    sns.set_palette("colorblind")
    sns.lineplot(
        x="Zeta",
        y="Runtime (s)",
        data=df,
        hue="Method",
        hue_order=[
            "Llama-2 (7B)",
            "Llama-2 (13B)",
            "Llama-2 (70B)",
            "Round Robin",
            "Random",
            "Online",
        ],
        legend=False,
    )
    plt.xlabel("Zeta")
    plt.xlim(0, 1)
    plt.savefig(
        f"routing-online-awareness-runtime-{gamma_K['Llama-2 (7B)']}-{gamma_K['Llama-2 (13B)']}-{gamma_K['Llama-2 (70B)']}.pdf",
        bbox_inches="tight",
    )

    plt.clf()
    plt.figure(figsize=(6, 5))
    sns.set(style="whitegrid", context="talk")
    sns.set_palette("colorblind")
    sns.lineplot(
        x="Zeta",
        y="Num Queries Llama-2 (7B)",
        data=df,
        marker="o",
        markersize=10,
        label="Llama-2 (7B)",
    )
    sns.lineplot(
        x="Zeta",
        y="Num Queries Llama-2 (13B)",
        data=df,
        marker="o",
        markersize=10,
        label="Llama-2 (13B)",
    )
    sns.lineplot(
        x="Zeta",
        y="Num Queries Llama-2 (70B)",
        data=df,
        marker="o",
        markersize=10,
        label="Llama-2 (70B)",
    )
    plt.xlabel("Zeta")
    plt.ylabel("Number of Queries Assigned")
    plt.legend(
        bbox_to_anchor=(0.5, -0.25),
        loc="center",
        ncol=3,
        frameon=False,
    )
    plt.savefig(
        f"routing-online-awareness-allocation-{gamma_K['Llama-2 (7B)']}-{gamma_K['Llama-2 (13B)']}-{gamma_K['Llama-2 (70B)']}.pdf",
        bbox_inches="tight",
    )

    plt.figure(figsize=(6, 5))
    sns.set(style="whitegrid", context="talk", font_scale=1.2)
    sns.set_palette("colorblind")
    sns.lineplot(
        x="Zeta",
        y="Average Wait Time (s)",
        data=df,
        hue="Method",
        hue_order=[
            "Round Robin",
            "Random",
            "Online",
        ],
        # legend=False,
    )
    plt.xlabel("Zeta")
    plt.xlim(0, 1)
    plt.ylabel("Average Wait Time (s)")
    plt.legend(
        bbox_to_anchor=(1, 0.5),
        loc="center left",
        frameon=False,
    )
    plt.savefig(
        f"routing-online-awareness-wait-time-{gamma_K['Llama-2 (7B)']}-{gamma_K['Llama-2 (13B)']}-{gamma_K['Llama-2 (70B)']}.pdf",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
