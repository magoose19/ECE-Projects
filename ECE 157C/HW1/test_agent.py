from __future__ import annotations

from pathlib import Path

import pandas as pd

from agent import run_agent


def get_dataset_config(base_dir: Path, choice: str) -> tuple[str, list[str], Path] | None:
    normalized_choice = choice.strip().lower()

    if normalized_choice == "housing data":
        return (
            str(base_dir / "housing.csv"),
            [
                "What is the average median house value across the dataset?",
                "Which ocean proximity category has the highest average median house value?",
                "What is the minimum, maximum, and median of median house values?",
                "How does median income vary across different ocean proximity categories?",
                "Which geographical areas (based on latitude and longitude ranges) have the highest average house prices?",
                "How does population density (population per household) relate to median house values?",
                "Identify the top 5 most expensive geographical areas and explain the key factors contributing to their high prices.",
                "Find coastal areas where house prices are relatively low despite proximity to the ocean. What factors might explain this?",
                "Find areas where house prices are significantly higher or lower than expected given median income. Explain possible reasons for these deviations.",
            ],
            base_dir / "housing_results.csv",
        )

    if normalized_choice == "plant data":
        return (
            str(base_dir / "plant_growth_data.csv"),
            [
                "What is the average Sunlight Hours across the dataset?",
                "Which Fertilizer Type is associated with the highest average Temperature?",
                "What are the minimum, maximum, and median values of Humidity?",
                "How does average Sunlight Hours vary across different Soil Types?",
                "Which combination of Water Frequency and Fertilizer Type has the highest average Sunlight Hours?",
                "How does Temperature relate to Humidity across the dataset?",
                "Identify the top 5 growing conditions (based on Soil Type, Fertilizer Type, and Water Frequency) with the highest average Sunlight Hours and explain the key environmental factors present in those conditions.",
                "Find cases where plants receive high Sunlight Hours but have low Humidity. What Soil Type and Temperature are most common in these cases?",
                "Identify plants with similar Soil Type, Water Frequency, and Fertilizer Type but significantly different Temperature and Humidity values. What patterns emerge?",
            ],
            base_dir / "plant_results.csv",
        )

    return None


def run_examples() -> None:
    base_dir = Path(__file__).resolve().parent
    rows: list[dict[str, str]] = []

    print("Which dataset would you like to evaluate?")
    print("1. housing data")
    print("2. plant data")
    choice = input("Enter your choice: ").strip().lower()

    if choice == "1":
        choice = "housing data"
    elif choice == "2":
        choice = "plant data"

    dataset_config = get_dataset_config(base_dir, choice)
    if dataset_config is None:
        print("Invalid choice. Please enter 'housing data' or 'plant data'.")
        return

    csv_path, examples, results_csv_path = dataset_config

    for index, question in enumerate(examples, start=1):
        result = run_agent(question=question, csv_path=csv_path)
        output_block = "\n".join(
            [
                f"Question: {question}",
                f"Answer: {result['final_answer']}",
                f"Status: {result['evaluation']}",
                "Generated Code:",
                result["generated_code"] or "[no code generated]",
            ]
        )
        rows.append(
            {
                "question_number": f"Q{index}",
                "output": output_block,
            }
        )
        print(f"\nExample {index}")
        print(output_block)

    columns = ["question_number", "output"]

    if results_csv_path.exists():
        existing_df = pd.read_csv(results_csv_path, dtype=str).fillna("")
        trial_count = existing_df["question_number"].astype(str).str.startswith("--- Trial ").sum()
    else:
        existing_df = pd.DataFrame(columns=columns)
        trial_count = 0

    separator_row = {
        "question_number": f"--- Trial {trial_count + 1} ---",
        "output": "",
    }
    new_rows_df = pd.DataFrame([separator_row, *rows], columns=columns)
    results_df = pd.concat([existing_df, new_rows_df], ignore_index=True)
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nSaved results to: {results_csv_path}")


if __name__ == "__main__":
    run_examples()
