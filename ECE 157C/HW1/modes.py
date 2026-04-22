from __future__ import annotations

from textwrap import dedent

DEFAULT_MODEL = "gpt-5.4-mini"
MAX_RETRIES = 3
CSV_PREVIEW_ROWS = 5

SYSTEM_PROMPT = dedent(
    """
    You are building Python data-analysis code for CSV question answering.
    Write clear, correct pandas code.

    Rules:
    - Return only raw Python code. Do not use markdown fences.
    - Read the CSV from the variable `csv_path`.
    - Use pandas as `pd`.
    - The code must assign the final computed answer to a variable named `FINAL_RESULT`.
    - The code may create helper variables, but `FINAL_RESULT` is required.
    - Prefer deterministic code and avoid unnecessary printing.
    - If a column name contains spaces or symbols, handle it correctly.
    - If the question asks for a count, aggregate, comparison, or filtered subset, compute it directly.
    """
).strip()

CODE_GEN_TEMPLATE = dedent(
    """
    Question:
    {question}

    CSV path:
    {csv_path}

    CSV schema summary:
    {dataset_summary}

    Previous attempt count:
    {attempt}

    Feedback from the previous attempt:
    {feedback}

    Generate Python code that answers the question from the CSV.
    Remember to assign the final answer to `FINAL_RESULT`.
    """
).strip()

EVALUATION_PROMPT = dedent(
    """
    You are evaluating whether a CSV analysis result correctly answers a user's question.

    Question:
    {question}

    CSV schema summary:
    {dataset_summary}

    Generated code:
    {generated_code}

    Execution result object:
    {execution_result}

    Decide whether the answer is good enough to return to the user.
    Mark FAIL if:
    - the code crashed
    - FINAL_RESULT is missing or null without justification
    - the result obviously does not answer the question
    - the code appears to use the wrong columns or wrong aggregation

    Respond as JSON with keys:
    - evaluation: PASS or FAIL
    - reasoning: short explanation
    """
).strip()

FINAL_ANSWER_PROMPT = dedent(
    """
    Write a concise natural-language answer for the user.

    Question:
    {question}

    Execution result object:
    {execution_result}

    Keep the answer grounded in the computed result. If the result contains a scalar,
    state it directly. If it contains a table-like structure, summarize the key takeaway.
    """
).strip()
