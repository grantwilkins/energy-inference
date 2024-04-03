import pandas as pd

df_falcon_palmetto = pd.read_csv("falcon-7b-palmetto-v100-1.csv")
df_mistral_palmetto = pd.read_csv("Mistral-7B-v0.1-palmetto-v100-1.csv")
df_llama_palmetto = pd.read_csv("Llama-2-7b-chat-hf-palmetto-v100-1.csv")

df_falcon_palmetto["GPU Energy (J)"] = (df_falcon_palmetto["GPU Energy (uJ)"]) / 1e6
df_mistral_palmetto["GPU Energy (J)"] = (df_mistral_palmetto["GPU Energy (uJ)"]) / 1e6
df_llama_palmetto["GPU Energy (J)"] = (df_llama_palmetto["GPU Energy (uJ)"]) / 1e6

df_falcon_swing = pd.read_csv("falcon-7b-argonne-swing-1.csv")
df_mistral_swing = pd.read_csv("Mistral-7B-v0.1-argonne-swing-1.csv")
df_llama_swing = pd.read_csv("Llama-2-7b-chat-hf-argonne-swing-1.csv")

df_falcon_swing["GPU Energy (J)"] = (df_falcon_swing["GPU Energy (mJ)"]) / 1e3
df_mistral_swing["GPU Energy (J)"] = (df_mistral_swing["GPU Energy (mJ)"]) / 1e3
df_llama_swing["GPU Energy (J)"] = (df_llama_swing["GPU Energy (mJ)"]) / 1e3

# df_falcon_mac = pd.read_csv("falcon-7b-M1-pro.csv")
# df_mistral_mac = pd.read_csv("Mistral-7B-v0.1-M1-Pro.csv")
df_llama_mac = pd.read_csv("Llama-2-7b-chat-hf-M1-Pro.csv")

df_falcon_palmetto.drop(
    [
        "Idle Package-0 Energy (uJ)",
        "Idle Package-1 Energy (uJ)",
        "CPU Package-0 Energy (uJ)",
        "CPU Package-1 Energy (uJ)",
        "GPU Energy (uJ)",
        "CPU Core",
    ],
    axis=1,
    inplace=True,
)
