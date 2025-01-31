import os
import csv
import datetime
from openwebui_model import OpenWebUIModel
from deepeval.benchmarks import GSM8K, DROP, ARC, BoolQ, LogiQA, BigBenchHard
from deepeval.benchmarks import MMLU
from deepeval.benchmarks.mmlu.task import MMLUTask
from deepeval.benchmarks.tasks import BigBenchHardTask

# Load OpenWebUI model
model = OpenWebUIModel()

# Define a subset limit (e.g., evaluate only 10 examples per benchmark for speed)
SUBSET_SIZE = 10  # Adjust this for faster/slower evaluation

# Define BBH tasks to focus on complex reasoning
BBH_TASKS = [
    # BigBenchHardTask.BOOLEAN_EXPRESSIONS,
    # BigBenchHardTask.CAUSAL_JUDGEMENT,
    BigBenchHardTask.DATE_UNDERSTANDING,
    BigBenchHardTask.LOGICAL_DEDUCTION_THREE_OBJECTS,
    BigBenchHardTask.MULTISTEP_ARITHMETIC_TWO,
    BigBenchHardTask.WORD_SORTING
]

# Define full benchmark suite
benchmarks = {
    # "MMLU": MMLU(
    #     tasks=[
    #         MMLUTask.HIGH_SCHOOL_COMPUTER_SCIENCE,
    #         # MMLUTask.ASTRONOMY,
    #         # MMLUTask.ELECTRICAL_ENGINEERING,
    #         # MMLUTask.PROFESSIONAL_MEDICINE
    #     ],
    #     n_shots=3,  # Few-shot learning
    #     n_problems_per_task=SUBSET_SIZE
    # ),
    # "GSM8K": GSM8K(n_problems=SUBSET_SIZE, ),
    # "DROP": DROP(n_problems_per_task=SUBSET_SIZE, ),
    "BIGBenchHard": BigBenchHard(
        tasks=BBH_TASKS,
        n_shots=3,  # Uses few-shot learning (0-3 shots)
        enable_cot=True,  # Enables Chain-of-Thought reasoning
        n_problems_per_task=SUBSET_SIZE, 
    ),
    # "ARC": ARC(n_problems=SUBSET_SIZE),
    # "BoolQ": BoolQ(n_problems=SUBSET_SIZE),
    # "LogiQA": LogiQA(n_problems_per_task=SUBSET_SIZE )
}

# Ensure CSV file is stored in the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of this script
csv_file = os.path.join(script_dir, "full_benchmark_results.csv")  # Save CSV in the same directory
headers = ["Timestamp", "Model Name", "Overall Score", "Samples per Benchmark"] + list(benchmarks.keys())

# Get current timestamp
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Collect benchmark scores
benchmark_scores = {}

for name, benchmark in benchmarks.items():
    print(f"🚀 Running subset benchmark: {name} with {SUBSET_SIZE} samples...")
   
    benchmark.evaluate(model=model)
    benchmark_scores[name] = benchmark.overall_score

# Calculate overall average score
overall_score = sum(benchmark_scores.values()) / len(benchmark_scores)

# Create a row with results
csv_row = [
    current_time,
    model.get_model_name(),
    overall_score,
    SUBSET_SIZE
] + [benchmark_scores.get(name, "N/A") for name in benchmarks.keys()]

# Append results to CSV file
file_exists = os.path.isfile(csv_file)

with open(csv_file, mode="a", newline="") as file:
    writer = csv.writer(file)

    # Write headers if the file does not exist
    if not file_exists:
        writer.writerow(headers)

    # Append new row
    writer.writerow(csv_row)

print(f"\n✅ Full benchmark completed. Results saved to {csv_file}.")