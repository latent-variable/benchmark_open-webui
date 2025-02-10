import os
import csv
import time
import datetime
from openwebui_model import OpenWebUIModel
from deepeval.benchmarks import GSM8K, DROP, ARC, BoolQ, LogiQA, BigBenchHard
from deepeval.benchmarks.modes import ARCMode
from deepeval.benchmarks import MMLU
from deepeval.benchmarks.mmlu.task import MMLUTask
from deepeval.benchmarks.tasks import BigBenchHardTask

# Define a subset limit (e.g., evaluate only 10 examples per benchmark for speed)
SUBSET_SIZE = 100 # Adjust this for faster/slower evaluation

# Define BBH tasks to focus on complex reasoning
BBH_TASKS = [
    BigBenchHardTask.BOOLEAN_EXPRESSIONS,
    BigBenchHardTask.CAUSAL_JUDGEMENT,
    BigBenchHardTask.DATE_UNDERSTANDING,
    BigBenchHardTask.LOGICAL_DEDUCTION_THREE_OBJECTS,
    BigBenchHardTask.MULTISTEP_ARITHMETIC_TWO,
]

# Define full benchmark suite
benchmarks = {
    "ARC": ARC(n_problems=SUBSET_SIZE, n_shots=0, mode=ARCMode.CHALLENGE),
    "GSM8K": GSM8K(n_problems=SUBSET_SIZE,n_shots=0,enable_cot=False)
}

# Define model names and whether CoT (Chain-of-Thought) is enabled
model_names = [
    # ("llama3.1:latest", False),
    # ("llama3.1:latest", True),
    ("deepseek_r1_reasoner.Reasoning_Effort_1/deepseek-r1:32b", True),
    ("deepseek_r1_reasoner_2.Reasoning_Effort_2/deepseek-r1:32b", True),
    ("deepseek_r1_reasoner_3.Reasoning_Effort_3/deepseek-r1:32b", True),
    ("deepseek_r1_reasoner_4.Reasoning_Effort_4/deepseek-r1:32b", True),
]

for benchmark_name, benchmark in benchmarks.items():
    for model_name, CoT in model_names:

        # Initialize OpenWebUIModel
        model = OpenWebUIModel(model=model_name, enable_cot=CoT)

        print(f"🚀 Running subset benchmark: {benchmark_name} with {SUBSET_SIZE} samples...")

        # Start timing the evaluation
        start_time = time.time()

        # Run the evaluation
        benchmark.evaluate(model=model)

        # End timing the evaluation
        end_time = time.time()
        elapsed_time = end_time - start_time
        benchmark_time = elapsed_time  # Store elapsed time

        benchmark_score = benchmark.overall_score
        print(f"✅ Benchmark '{benchmark_name}' completed in {elapsed_time:.2f} seconds.")

        # Calculate overall average score
        overall_score = benchmark_score

        # Ensure CSV file is stored in the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of this script
        csv_file = os.path.join(script_dir, "full_benchmark_results.csv")  # Save CSV in the same directory
        headers = ["Benchmark", "Timestamp", "Model Name", "CoT Enabled", "Overall Score", "Samples per Benchmark", "Total Evaluation Time (s)"]

        # Get current timestamp
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create a row with results
        csv_row = [
            benchmark_name,
            current_time,
            model.get_model_name(),
            CoT,  # Include CoT status
            overall_score,
            SUBSET_SIZE,
            benchmark_time# Total evaluation time
        ] 

        # Append results to CSV file
        file_exists = os.path.isfile(csv_file)

        with open(csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)

            # Write headers if the file does not exist
            if not file_exists:
                writer.writerow(headers)

            # Append new row
            writer.writerow(csv_row)

print(f"\n✅ {model.get_model_name()} benchmark completed. Results saved to {csv_file}.")