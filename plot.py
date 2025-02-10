import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create the DataFrame
df_32b = pd.read_csv("./benchmark_open-webui/full_benchmark_results_32b.csv")

# Process the data
df_32b["Timestamp"] = pd.to_datetime(df_32b["Timestamp"])
df_32b["Total Evaluation Time (min)"] = df_32b["Total Evaluation Time (s)"] / 60

# Refine model names
def refined_model_name(name, cot_enabled):
    if "Reasoning_Effort" in name:
        effort_number = name.split("Reasoning_Effort_")[-1].split("/")[0]
        return f"Reasoning Effort {effort_number}"
    return name

df_32b["Refined Model Name"] = df_32b.apply(lambda row: refined_model_name(row["Model Name"], row["CoT Enabled"]), axis=1)

# Set up benchmarks and palette
benchmarks_32b = df_32b["Benchmark"].unique()
palette_32b = sns.color_palette("pastel", df_32b["Refined Model Name"].nunique())

# Generate plots
for benchmark in benchmarks_32b:
    benchmark_df = df_32b[df_32b["Benchmark"] == benchmark]

    # Overall Score Plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=benchmark_df, x="Refined Model Name", y="Overall Score", palette=palette_32b)
    plt.title(f"Overall Score by Model - {benchmark} Benchmark (32B)")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Model Name")
    plt.ylabel("Overall Score")

    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')

    plt.show()

    # Evaluation Time Plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=benchmark_df, x="Refined Model Name", y="Total Evaluation Time (min)", palette=palette_32b)
    plt.title(f"Total Evaluation Time (minutes) by Model - {benchmark} Benchmark (32B)")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Model Name")
    plt.ylabel("Evaluation Time (min)")

    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')

    plt.show()