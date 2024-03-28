import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import numpy as np

df = pd.read_csv("Llama-2-7b-chat-hf-M1-Pro.csv")
df_mistral = pd.read_csv("Mistral-7B-v0.1-M1-Pro.csv")

# df = pd.concat([df, df_mistral])

plt.figure(figsize=(4, 3))
sns.set_theme(style="whitegrid")
sns.set_context("paper")

print(df)
df = df[df["Operation"].str.contains("inference")]
print(df)
df["Throughput (tokens/s)"] = df["Output Token Size"] / df["Runtime (s)"]
df["Total Energy (J)"] = df["GPU Energy (J)"] + df["CPU Energy (J)"]
df["Energy per Token (J)"] = df["Total Energy (J)"] / df["Output Token Size"]

# sns.histplot(
#     data=df,
#     hue="Model Name",
#     x="Runtime (s)",
# )
# plt.show()

# sns.histplot(
#     data=df,
#     x="Throughput (tokens/s)",
#     hue="Model Name",
#     stat="density",
# )
# plt.show()


# sns.histplot(
#     data=df,
#     x="Total Energy (J)",
# )
# plt.show()

# sns.histplot(
#     data=df,
#     x="Energy per Token (J)",
#     hue="Model Name",
#     stat="density",
# )
# plt.show()
confidence_level = 0.95
runtimes = df["Runtime (s)"].to_numpy()
mean_runtime = np.mean(runtimes)
std_err = stats.sem(runtimes)
z_critical = stats.norm.ppf((1 + confidence_level) / 2)
ci_half_width = z_critical * std_err


# sns.lineplot(
#     data=df,
#     x="Output Token Size",
#     y="Energy per Token (J)",
#     hue="Model Name",
# )
# plt.show()
