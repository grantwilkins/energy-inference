import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

token_count = 200
# Load the CSV data
df_1gpu = pd.read_csv("llama2-7b-cuda-1.csv", sep=";")
df_1gpu["GPU Count"] = 1
df_2gpu = pd.read_csv("llama2-7b-cuda-2.csv", sep=";")
df_2gpu["GPU Count"] = 2
df_4gpu = pd.read_csv("llama2-7b-cuda-4.csv", sep=";")
df_4gpu["GPU Count"] = 4
df_8gpu = pd.read_csv("llama2-7b-cuda-8.csv", sep=";")
df_8gpu["GPU Count"] = 8
df = pd.concat([df_1gpu, df_2gpu, df_4gpu, df_8gpu]).fillna(0)
df["Total GPU Energy (J)"] = (
    df["nvidia_gpu_0"]
    + df["nvidia_gpu_1"]
    + df["nvidia_gpu_2"]
    + df["nvidia_gpu_3"]
    + df["nvidia_gpu_4"]
    + df["nvidia_gpu_5"]
    + df["nvidia_gpu_6"]
    + df["nvidia_gpu_7"]
) * 0.001
df["Token Rate (tokens/s)"] = token_count / df["duration"]
df["Event"] = df["tag"].str.capitalize()
df["Model"] = "Llama2-7b"

df_inference = df[df["Event"] == "Inference"]
print(df_inference.groupby("GPU Count")["Token Rate (tokens/s)"].mean())

sns.set_theme(style="whitegrid")
sns.set_context("paper")
sns.barplot(
    x="Event",
    y="Total GPU Energy (J)",
    hue="GPU Count",
    data=df,
)
plt.legend(loc="best", title="GPU Count")
plt.xlabel("Event")
plt.ylabel("Energy (J)")
plt.tight_layout()
plt.savefig("llama2-7b-cuda-barchart.pdf")

df.to_csv("llama2-7b-cuda.csv", sep=",")

df["Energy per Token (J)"] = df["Total GPU Energy (J)"] / token_count
print(df.groupby("GPU Count")["Total GPU Energy (J)"].mean())
print(df.groupby("GPU Count")["Energy per Token (J)"].mean())
