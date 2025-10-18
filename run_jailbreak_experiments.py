"""
Experiment script to test jailbreak scenarios and track attack generation iterations.

This script:
1. Takes JAILBREAK_SCENARIOS list
2. Parses each scenario through parse_user_input to get behavior and reference
3. Uses test_jailbreak_with_iterations to generate attack prompts
4. Records: final attack prompt, conversation history, number of iterations
"""

import json
import fire
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional

from logger import logger
from prompts import JAILBREAK_SCENARIOS
from parsing_agent import parse_user_input
from jailbreak_agent import test_jailbreak_with_iterations


def run_experiments(
    jailbreak_model_name: str = "google/gemini-2.5-flash-lite",
    attack_model_name: str = "google/gemini-2.5-flash-lite",
    judge_model_name: str = "google/gemini-2.5-flash-lite",
    translator_model_name: str = "google/gemini-2.5-flash-lite",
    max_attempts: int = 5,
    output_dir: str = "./jailbreak-experiments",
    limit: Optional[int] = None
):
    """
    Run jailbreak experiments across all scenarios.

    Args:
        jailbreak_model_name: Model being tested for jailbreaks
        attack_model_name: Model generating attack prompts
        judge_model_name: Model judging success/failure
        translator_model_name: Model parsing scenarios into behaviors/references
        max_attempts: Maximum refinement attempts per scenario
        output_dir: Directory to save results
        limit: Optional limit on number of scenarios to test (for debugging)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = Path(output_dir) / timestamp
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting jailbreak experiments at {timestamp}")
    logger.info(f"Testing {len(JAILBREAK_SCENARIOS) if limit is None else limit} scenarios")
    logger.info(f"Jailbreak model: {jailbreak_model_name}")
    logger.info(f"Output directory: {output_path}")

    results = []
    scenarios_to_test = JAILBREAK_SCENARIOS[:limit] if limit else JAILBREAK_SCENARIOS

    for idx, scenario in enumerate(scenarios_to_test, 1):
        logger.info("=" * 100)
        logger.info(f"Scenario {idx}/{len(scenarios_to_test)}")
        logger.info(f"Scenario text: {scenario}")
        logger.info("=" * 100)

        try:
            # Step 1: Parse scenario into behavior and reference
            logger.info("Parsing scenario...")
            behavior, reference = parse_user_input(
                user_input=scenario,
                translator_model_name=translator_model_name
            )

            logger.info(f"Parsed behavior: {behavior}")
            logger.info(f"Parsed reference: {reference}")

            # Handle empty parsing results
            if not behavior or not reference:
                logger.warning("Failed to parse scenario - skipping")
                results.append({
                    "scenario_index": idx,
                    "scenario": scenario,
                    "behavior": behavior,
                    "reference": reference,
                    "success": False,
                    "num_iterations": 0,
                    "final_attack_prompt": "",
                    "conversation_history": [],
                    "target_response": "",
                    "judge_verdict": "",
                    "error": "Failed to parse scenario"
                })
                continue

            # Step 2: Run jailbreak attack with iteration tracking
            logger.info("Running jailbreak attack...")
            attack_prompt, num_iterations, conversation_history, target_response, judge_verdict = test_jailbreak_with_iterations(
                behavior=behavior,
                reference=reference,
                attack_llm_name=attack_model_name,
                jailbreak_model_name=jailbreak_model_name,
                judge_model_name=judge_model_name,
                max_attempts=max_attempts
            )

            success = bool(attack_prompt)  # Empty string means failure

            # Step 3: Record results
            result = {
                "scenario_index": idx,
                "scenario": scenario,
                "behavior": behavior,
                "reference": reference,
                "success": success,
                "num_iterations": num_iterations,
                "final_attack_prompt": attack_prompt,
                "conversation_history": conversation_history,
                "target_response": target_response,
                "judge_verdict": judge_verdict,
                "error": None
            }
            results.append(result)

            logger.info(f"Result: {'SUCCESS' if success else 'FAILURE'} after {num_iterations} iterations")

        except Exception as e:
            logger.error(f"Error processing scenario {idx}: {e}")
            results.append({
                "scenario_index": idx,
                "scenario": scenario,
                "behavior": "",
                "reference": "",
                "success": False,
                "num_iterations": 0,
                "final_attack_prompt": "",
                "conversation_history": [],
                "target_response": "",
                "judge_verdict": "",
                "error": str(e)
            })

    # Save results
    logger.info("=" * 100)
    logger.info("Saving results...")

    # Prepare JSON output with model metadata
    output_data = {
        "metadata": {
            "timestamp": timestamp,
            "jailbreak_model_name": jailbreak_model_name,
            "attack_model_name": attack_model_name,
            "judge_model_name": judge_model_name,
            "translator_model_name": translator_model_name,
            "max_attempts": max_attempts,
            "total_scenarios": len(results),
            "successful_jailbreaks": sum(1 for r in results if r["success"]),
        },
        "results": results
    }

    # Save as JSON (includes full conversation history)
    json_path = output_path / "results.json"
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"Saved full results to {json_path}")

    # Save as CSV (summary without conversation history)
    csv_data = [{
        "scenario_index": r["scenario_index"],
        "scenario": r["scenario"],
        "behavior": r["behavior"],
        "reference": r["reference"],
        "success": r["success"],
        "num_iterations": r["num_iterations"],
        "final_attack_prompt": r["final_attack_prompt"],
        "target_response": r["target_response"],
        "judge_verdict": r["judge_verdict"],
        "error": r["error"]
    } for r in results]

    csv_path = output_path / "results.csv"
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved summary to {csv_path}")

    # Print summary statistics
    total = len(results)
    successful = sum(1 for r in results if r["success"])
    failed = total - successful
    avg_iterations = sum(r["num_iterations"] for r in results if r["success"]) / successful if successful > 0 else 0

    logger.info("=" * 100)
    logger.info("EXPERIMENT SUMMARY")
    logger.info(f"Total scenarios: {total}")
    logger.info(f"Successful jailbreaks: {successful} ({successful/total*100:.1f}%)")
    logger.info(f"Failed jailbreaks: {failed} ({failed/total*100:.1f}%)")
    logger.info(f"Average iterations (successful): {avg_iterations:.2f}")
    logger.info("=" * 100)

    return results


if __name__ == "__main__":
    fire.Fire(run_experiments)
