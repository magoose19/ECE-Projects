from __future__ import annotations

import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from typing_extensions import Literal, TypedDict

from modes import (
    CODE_GEN_TEMPLATE,
    CSV_PREVIEW_ROWS,
    DEFAULT_MODEL,
    EVALUATION_PROMPT,
    FINAL_ANSWER_PROMPT,
    MAX_RETRIES,
    SYSTEM_PROMPT,
)


class AgentState(TypedDict, total=False):
    question: str
    csv_path: str
    dataset_summary: str
    generated_code: str
    execution_code: dict[str, Any]
    evaluation: Literal["PASS", "FAIL"]
    evaluation_reason: str
    final_answer: str
    attempt: int
    last_feedback: str


def _load_environment(csv_path: str) -> None:
    csv_dir = Path(csv_path).resolve().parent
    candidate_paths = [
        csv_dir / ".env",
        Path(__file__).resolve().parent / ".env",
    ]
    for env_path in candidate_paths:
        if env_path.exists():
            load_dotenv(env_path, override=False)
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY was not found. Add it to a .env file in the project folder."
        )


def _build_model() -> ChatOpenAI:
    return ChatOpenAI(model=DEFAULT_MODEL, temperature=0)


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            else:
                text_parts.append(str(item))
        return "\n".join(part for part in text_parts if part).strip()
    return str(content)


def _clean_code(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        records = value.head(20).to_dict(orient="records")
        return _json_safe(records)
    if isinstance(value, pd.Series):
        return _json_safe(value.head(20).to_dict())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _summarize_csv(state: AgentState) -> AgentState:
    csv_path = Path(state["csv_path"]).resolve()
    preview_df = pd.read_csv(csv_path, nrows=CSV_PREVIEW_ROWS)
    full_df = pd.read_csv(csv_path)

    schema_rows = []
    for column, dtype in full_df.dtypes.items():
        nulls = int(full_df[column].isna().sum())
        sample_values = full_df[column].dropna().astype(str).head(3).tolist()
        schema_rows.append(
            f"- {column} ({dtype}), nulls={nulls}, sample={sample_values}"
        )

    summary = "\n".join(
        [
            f"Rows: {len(full_df)}",
            f"Columns: {len(full_df.columns)}",
            "Column details:",
            *schema_rows,
            "",
            "Preview rows:",
            preview_df.to_csv(index=False).strip(),
        ]
    )
    return {
        "dataset_summary": summary,
        "attempt": 0,
        "last_feedback": "No previous attempt.",
    }


def _generate_code(state: AgentState) -> AgentState:
    model = _build_model()
    prompt = CODE_GEN_TEMPLATE.format(
        question=state["question"],
        csv_path=state["csv_path"],
        dataset_summary=state["dataset_summary"],
        attempt=state.get("attempt", 0) + 1,
        feedback=state.get("last_feedback", "No previous attempt."),
    )
    response = model.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )
    return {
        "generated_code": _clean_code(_message_text(response.content)),
        "attempt": state.get("attempt", 0) + 1,
    }


def _execute_code(state: AgentState) -> AgentState:
    csv_path = str(Path(state["csv_path"]).resolve())
    local_vars: dict[str, Any] = {}
    stdout_buffer = io.StringIO()

    try:
        with redirect_stdout(stdout_buffer):
            exec(
                state["generated_code"],
                {"__builtins__": __builtins__, "pd": pd, "csv_path": csv_path},
                local_vars,
            )
        final_result = local_vars.get("FINAL_RESULT")
        execution_code = {
            "success": "FINAL_RESULT" in local_vars,
            "result": _json_safe(final_result),
            "stdout": stdout_buffer.getvalue().strip(),
            "error": None if "FINAL_RESULT" in local_vars else "FINAL_RESULT was not assigned.",
        }
    except Exception as exc:  # noqa: BLE001
        execution_code = {
            "success": False,
            "result": None,
            "stdout": stdout_buffer.getvalue().strip(),
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {"execution_code": execution_code}


def _evaluate_result(state: AgentState) -> AgentState:
    execution = state["execution_code"]
    if not execution.get("success"):
        reason = execution.get("error") or "Execution failed."
        return {
            "evaluation": "FAIL",
            "evaluation_reason": reason,
            "last_feedback": reason,
        }

    model = _build_model()
    parser = JsonOutputParser()
    prompt = EVALUATION_PROMPT.format(
        question=state["question"],
        dataset_summary=state["dataset_summary"],
        generated_code=state["generated_code"],
        execution_result=json.dumps(state["execution_code"], indent=2),
    )
    response = model.invoke(
        [
            SystemMessage(content="Return only valid JSON."),
            HumanMessage(content=prompt),
        ]
    )

    try:
        parsed = parser.parse(_message_text(response.content))
        evaluation = parsed.get("evaluation", "FAIL")
        reasoning = parsed.get("reasoning", "No reasoning returned.")
    except Exception:  # noqa: BLE001
        evaluation = "FAIL"
        reasoning = f"Could not parse evaluator output: {_message_text(response.content)}"

    if evaluation not in {"PASS", "FAIL"}:
        evaluation = "FAIL"

    return {
        "evaluation": evaluation,
        "evaluation_reason": reasoning,
        "last_feedback": reasoning,
    }


def _should_retry(state: AgentState) -> str:
    if state.get("evaluation") == "PASS":
        return "answer"
    if state.get("attempt", 0) >= MAX_RETRIES:
        return "stop"
    return "retry"


def _write_final_answer(state: AgentState) -> AgentState:
    if state.get("evaluation") != "PASS":
        return {
            "final_answer": (
                "I could not confidently answer the question after multiple attempts. "
                f"Last evaluation feedback: {state.get('evaluation_reason', 'Unknown issue.')}"
            )
        }

    model = _build_model()
    prompt = FINAL_ANSWER_PROMPT.format(
        question=state["question"],
        execution_result=json.dumps(state["execution_code"], indent=2),
    )
    response = model.invoke([HumanMessage(content=prompt)])
    return {"final_answer": _message_text(response.content).strip()}


def _build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("summarize_csv", _summarize_csv)
    graph.add_node("generate_code", _generate_code)
    graph.add_node("execute_code", _execute_code)
    graph.add_node("evaluate_result", _evaluate_result)
    graph.add_node("write_final_answer", _write_final_answer)

    graph.set_entry_point("summarize_csv")
    graph.add_edge("summarize_csv", "generate_code")
    graph.add_edge("generate_code", "execute_code")
    graph.add_edge("execute_code", "evaluate_result")
    graph.add_conditional_edges(
        "evaluate_result",
        _should_retry,
        {
            "retry": "generate_code",
            "answer": "write_final_answer",
            "stop": "write_final_answer",
        },
    )
    graph.add_edge("write_final_answer", END)
    return graph.compile()


def run_agent(question: str, csv_path: str) -> dict[str, Any]:
    try:
        _load_environment(csv_path)
        app = _build_graph()
        result = app.invoke({"question": question, "csv_path": csv_path})
        return {
            "generated_code": result.get("generated_code", ""),
            "execution_code": result.get("execution_code", {}),
            "evaluation": result.get("evaluation", "FAIL"),
            "final_answer": result.get("final_answer", ""),
        }
    except Exception as exc:  # noqa: BLE001
        message = f"{type(exc).__name__}: {exc}"
        return {
            "generated_code": "",
            "execution_code": {
                "success": False,
                "result": None,
                "stdout": "",
                "error": message,
            },
            "evaluation": "FAIL",
            "final_answer": (
                "The agent could not complete the request because of an upstream error. "
                f"Details: {message}"
            ),
        }


if __name__ == "__main__":
    sample_csv = Path(__file__).resolve().parent / "housing.csv"
    sample_question = "What is the average median_house_value grouped by ocean_proximity?"
    output = run_agent(sample_question, str(sample_csv))
    print(f"Question: {sample_question}")
    print(f"Answer: {output['final_answer']}")
    print(f"Status: {output['evaluation']}")
    print("Generated Code:")
    print(output["generated_code"] or "[no code generated]")
