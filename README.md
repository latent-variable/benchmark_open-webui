# OpenWebUI Benchmarking Suite

This project provides a framework for benchmarking and evaluating OpenWebUI language models with support for both direct and Chain-of-Thought (CoT) reasoning. It utilizes DeepEval for systematic benchmarking across multiple datasets.

## Features
- **Support for CoT and Direct Prompting**: Evaluate models using either standard direct prompting or Chain-of-Thought reasoning.
- **Customizable Benchmarks**: Run benchmarks on popular datasets like ARC, GSM8K, DROP, and more.
- **Time Tracking**: Measure and log the runtime for each evaluation.
- **Structured Output**: Save results (including runtime, scores, and configuration) to a CSV file for easy analysis.

## Setup

### Prerequisites
- Python 3.8+
- The following Python libraries:
  - `requests`
  - `pydantic`
  - `deepeval`

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/openwebui_benchmark.git
   cd openwebui_benchmark
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure your OpenWebUI API is running locally on port `3000` (default).

4. Set up your API key in the code or environment variables or Create a .env file in the project directory::
   ```bash
   export OPENWEBUI_API_KEY=<your_api_key>
   ```

## Usage

### Running Benchmarks
Edit the `main.py` script to configure the models and benchmarks you want to evaluate. For example:

```python
model_names = [
    ("llama3.1:latest", False),  # Direct prompting
    ("llama3.1:latest", True),   # CoT prompting
    ...
]
```

Run the script:
```bash
python main.py
```

### Results
- Results will be saved to `full_benchmark_results.csv` in the project directory.
- The CSV includes:
  - Model name
  - CoT status (enabled/disabled)
  - Benchmark scores
  - Evaluation time per benchmark

### Example Output
```
Timestamp           Model Name                    CoT Enabled  Overall Score  Samples  Total Time (s)  ARC    ARC_Time
2025-02-07 12:00:00 llama3.1:latest              False        78.5           100      35.2            78.5   35.2
2025-02-07 12:15:00 deepseek_r1_reasoner_1:8b   True         85.7           100      40.1            85.7   40.1
```

### Adding New Benchmarks
To add a new benchmark:
1. Import it from `deepeval.benchmarks`.
2. Add it to the `benchmarks` dictionary in the script, e.g.:
   ```python
   from deepeval.benchmarks import BoolQ

   benchmarks = {
       "ARC": ARC(n_problems=100, n_shots=0, mode=ARCMode.CHALLENGE),
       "BoolQ": BoolQ(n_problems=100),
   }
   ```

## File Structure
```
openwebui_benchmark/
│
├── openwebui_model.py          # Core OpenWebUIModel class
├── main.py                     # Benchmarking script
├── requirements.txt            # Python dependencies
├── full_benchmark_results.csv  # Results file (generated after running)
└── README.md                   # Documentation
```

## Contributing
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-branch
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add new feature"
   ```
4. Push the branch:
   ```bash
   git push origin feature-branch
   ```
5. Create a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
For questions or support, please contact [lino.valdovinos.cs@gmail.com].
