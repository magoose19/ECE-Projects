# LangGraph CSV QA Agent

This project builds a small AI agent with LangGraph that:

- takes a natural-language question
- reads a CSV dataset
- generates Python code
- executes that code
- evaluates whether the result is acceptable
- retries code generation if evaluation fails
- returns a final natural-language answer

## Files

- `agent.py`: main LangGraph agent and `run_agent(question: str, csv_path: str) -> dict`
- `modes.py`: model settings and prompt templates
- `test_agent.py`: simple runner with example question/CSV pairs
- `requirements.txt`: Python dependencies

## Expected Output

`run_agent(...)` returns a dictionary with this structure:

```python
{
    "generated_code": str,
    "execution_code": object,
    "evaluation": "PASS" or "FAIL",
    "final_answer": str,
}
```

## Setup

1. Create or activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Put your OpenAI API key in `.env` in this folder:

```env
OPENAI_API_KEY=your_key_here
```

## Run the Agent

You can call the required function directly:

```python
from agent import run_agent

result = run_agent(
    question="What is the average median_house_value for each ocean_proximity category?",
    csv_path="housing.csv",
)

print(result)
```

Or run the example harness:

```bash
python test_agent.py
```

When prompted, type either:

- `housing data`
- `plant data`

You can also type `1` for housing or `2` for plant data.

### Dataset Options

`housing data`

- Uses `housing.csv`
- Saves results to `housing_results.csv`
- Runs questions about house values, ocean proximity, income, geographic price patterns, and pricing anomalies

`plant data`

- Uses `plant_growth_data.csv`
- Saves results to `plant_results.csv`
- Runs questions about plant height, fertilizer performance, soil effects, sunlight and watering combinations, temperature, and unusual growth outcomes

### Output Files

- Choosing `housing data` appends results to `housing_results.csv`
- Choosing `plant data` appends results to `plant_results.csv`

Each saved CSV contains:

- `question_number`
- `output`

The `output` column stores the same formatted block shown in the terminal for each question.

## Notes

- The agent uses `gpt-5.4-mini` as configured in `modes.py`.
- The `.env` file is loaded from the same directory as the CSV file.
- Generated code must set `FINAL_RESULT`, which is used by execution and evaluation.
- If evaluation fails, the graph retries code generation up to 3 attempts.
- If the OpenAI API returns an upstream error such as quota or billing issues, `run_agent(...)` returns a structured `FAIL` response instead of raising a crash.
