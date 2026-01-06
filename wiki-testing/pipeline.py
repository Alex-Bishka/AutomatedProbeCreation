"""
Agentic Deception Probe Pipeline - Orchestrator

This module orchestrates the full pipeline:
1. Load vetted template scenarios
2. Generate new scenarios autonomously
3. Quality check generated scenarios
4. Select SAE features for steering
5. Run steering execution
6. Evaluate results
7. Create deception probe
"""

import os
import json
import time
import uuid
from datetime import datetime
from typing import Optional
from pathlib import Path

from agents import (
    ScenarioCreatorAgent,
    ScenarioQualityJudge,
    FeatureSelectorAgent,
    EvaluationJudgeAgent,
    DEFAULT_MODEL,
    HIGH_QUALITY_MODEL,
    VETTED_CATEGORIES,
    get_agent,
    load_pipeline_config
)
from neuronpedia import steering_chat

# Paths
FRONTEND_DIR = Path(__file__).parent / "frontend"
SCENARIOS_FILE = FRONTEND_DIR / "scenarios.json"
PIPELINE_SCENARIOS_FILE = FRONTEND_DIR / "pipeline_scenarios.json"
PIPELINE_JOBS_FILE = FRONTEND_DIR / "pipeline_jobs.json"
REVIEW_QUEUE_FILE = FRONTEND_DIR / "review_queue.json"
PROBES_DIR = FRONTEND_DIR / "probes"
JOB_DETAILS_DIR = FRONTEND_DIR / "job_details"

# Ensure directories exist
PROBES_DIR.mkdir(exist_ok=True)
JOB_DETAILS_DIR.mkdir(exist_ok=True)


def extract_response(resp) -> str:
    """Extract the last assistant message content from a steering response."""
    if isinstance(resp, dict) and "chatTemplate" in resp:
        chat = resp["chatTemplate"]
        if isinstance(chat, list) and len(chat) > 0:
            return chat[-1].get("content", "")
    return str(resp) if resp else ""


def load_scenarios() -> list[dict]:
    """Load scenarios from the main scenarios file."""
    if SCENARIOS_FILE.exists():
        with open(SCENARIOS_FILE, 'r') as f:
            return json.load(f)
    return []


def load_vetted_templates() -> list[dict]:
    """Load only vetted template scenarios (Fear/Survival + Corporate Loyalty)."""
    scenarios = load_scenarios()
    return [s for s in scenarios if s.get("category") in VETTED_CATEGORIES]


def load_pipeline_scenarios() -> list[dict]:
    """Load pipeline-generated scenarios."""
    if PIPELINE_SCENARIOS_FILE.exists():
        with open(PIPELINE_SCENARIOS_FILE, 'r') as f:
            return json.load(f)
    return []


def save_pipeline_scenarios(scenarios: list[dict]):
    """Save pipeline-generated scenarios."""
    with open(PIPELINE_SCENARIOS_FILE, 'w') as f:
        json.dump(scenarios, f, indent=2)


def load_pipeline_jobs() -> list[dict]:
    """Load pipeline job records."""
    if PIPELINE_JOBS_FILE.exists():
        with open(PIPELINE_JOBS_FILE, 'r') as f:
            return json.load(f)
    return []


def save_pipeline_jobs(jobs: list[dict]):
    """Save pipeline job records."""
    with open(PIPELINE_JOBS_FILE, 'w') as f:
        json.dump(jobs, f, indent=2)


def load_review_queue() -> list[dict]:
    """Load the review queue."""
    if REVIEW_QUEUE_FILE.exists():
        with open(REVIEW_QUEUE_FILE, 'r') as f:
            return json.load(f)
    return []


def save_review_queue(queue: list[dict]):
    """Save the review queue."""
    with open(REVIEW_QUEUE_FILE, 'w') as f:
        json.dump(queue, f, indent=2)


def save_job_details(job_id: str, details: dict):
    """Save detailed job results to a separate file for transparency."""
    filepath = JOB_DETAILS_DIR / f"{job_id}.json"
    with open(filepath, 'w') as f:
        json.dump(details, f, indent=2)


def load_job_details(job_id: str) -> Optional[dict]:
    """Load detailed job results from file."""
    filepath = JOB_DETAILS_DIR / f"{job_id}.json"
    if filepath.exists():
        with open(filepath, 'r') as f:
            return json.load(f)
    return None


def add_to_review_queue(
    scenario: dict,
    default_response: str,
    steered_response: str,
    features: list[dict],
    evaluation: dict,
    job_id: str
):
    """Add an item to the review queue."""
    queue = load_review_queue()

    item = {
        "id": f"review_{uuid.uuid4().hex[:8]}",
        "created_at": datetime.now().isoformat(),
        "job_id": job_id,
        "scenario": scenario,
        "default_response": default_response,
        "steered_response": steered_response,
        "features": features,
        "judge_evaluation": evaluation,
        "status": "pending",
        "human_verdict": None,
        "notes": ""
    }

    queue.append(item)
    save_review_queue(queue)
    return item["id"]


class PipelineOrchestrator:
    """
    Orchestrates the full deception probe pipeline.

    The pipeline runs autonomously:
    1. Loads vetted template scenarios
    2. Generates new scenarios via ScenarioCreatorAgent
    3. Quality-checks via ScenarioQualityJudge
    4. Selects features via FeatureSelectorAgent
    5. Runs steering on all approved scenarios
    6. Evaluates results via EvaluationJudgeAgent
    7. Routes failures to human review queue
    8. Collects successes for probe training
    """

    def __init__(self, high_quality_mode: bool = False):
        """
        Initialize the pipeline.

        Args:
            high_quality_mode: If True, use google/gemini-2.5-pro instead of flash
        """
        self.high_quality_mode = high_quality_mode
        self.model = HIGH_QUALITY_MODEL if high_quality_mode else DEFAULT_MODEL

        # Initialize agents
        self.scenario_agent = ScenarioCreatorAgent(self.model)
        self.quality_judge = ScenarioQualityJudge(self.model)
        self.feature_agent = FeatureSelectorAgent(self.model)
        self.eval_judge = EvaluationJudgeAgent(self.model)

        # Job tracking
        self.job_id: Optional[str] = None
        self.job_status: dict = {}

    def _update_job_status(self, step: str, current: int = 0, total: int = 0, **kwargs):
        """Update job status."""
        self.job_status.update({
            "step": step,
            "current": current,
            "total": total,
            "updated_at": datetime.now().isoformat(),
            **kwargs
        })

        # Persist to file
        jobs = load_pipeline_jobs()
        for job in jobs:
            if job["id"] == self.job_id:
                job["status"] = "running"
                job["progress"] = self.job_status
                break
        save_pipeline_jobs(jobs)

    def _run_steering_experiments(
        self,
        scenarios: list[dict],
        selected_features: list[dict],
        target_model: str,
        test_strengths: list[int]
    ) -> list[dict]:
        """
        Test each feature individually at multiple strength levels.

        Args:
            scenarios: List of approved scenarios to test
            selected_features: Features to test (one per layer after dedup)
            target_model: Model to steer
            test_strengths: List of strength values to test (e.g., [2, 5, 10])

        Returns:
            List of experiment results, one per feature:
            [
                {
                    "feature": {...},
                    "strength_results": {
                        2: [{"scenario": {...}, "default": "...", "steered": "...", "error": None}, ...],
                        5: [...],
                        10: [...]
                    }
                },
                ...
            ]
        """
        experiments = []
        total_experiments = len(selected_features) * len(test_strengths) * len(scenarios)
        current_experiment = 0

        # Cache default responses per scenario (same across all features)
        default_responses_cache = {}

        for feature in selected_features:
            feature_experiment = {
                "feature": {
                    "layer": feature.get("layer", ""),
                    "index": feature.get("index", 0),
                    "description": feature.get("description", ""),
                    "cosine_similarity": feature.get("cosine_similarity", 0),
                    "concept": feature.get("concept", "")
                },
                "strength_results": {}
            }

            for strength in test_strengths:
                steering_config = [{
                    "modelId": target_model,
                    "layer": feature.get("layer", ""),
                    "index": feature.get("index", 0),
                    "strength": strength
                }]

                scenario_results = []
                for scenario in scenarios:
                    current_experiment += 1
                    self._update_job_status(
                        "steering_experiments",
                        current_experiment,
                        total_experiments,
                        feature_layer=feature.get("layer", ""),
                        feature_index=feature.get("index", 0),
                        current_strength=strength
                    )

                    scenario_key = scenario.get("name", str(id(scenario)))
                    messages = scenario.get("messages", [])

                    try:
                        # Get default response (cached)
                        if scenario_key in default_responses_cache:
                            default_text = default_responses_cache[scenario_key]
                            # Only need to get steered response
                            _, steered_resp = steering_chat(
                                messages,
                                steering_config,
                                model=target_model
                            )
                            steered_text = extract_response(steered_resp)
                        else:
                            # First time - get both
                            default_resp, steered_resp = steering_chat(
                                messages,
                                steering_config,
                                model=target_model
                            )
                            default_text = extract_response(default_resp)
                            steered_text = extract_response(steered_resp)
                            default_responses_cache[scenario_key] = default_text

                        scenario_results.append({
                            "scenario_name": scenario.get("name", "Unknown"),
                            "scenario": scenario,
                            "default": default_text,
                            "steered": steered_text,
                            "error": None
                        })
                    except Exception as e:
                        scenario_results.append({
                            "scenario_name": scenario.get("name", "Unknown"),
                            "scenario": scenario,
                            "default": default_responses_cache.get(scenario_key, ""),
                            "steered": "",
                            "error": str(e)
                        })

                feature_experiment["strength_results"][strength] = scenario_results

            experiments.append(feature_experiment)

        return experiments

    def run(
        self,
        num_scenarios: int = 10,
        target_model: str = "llama3.1-8b-it",
        max_features_per_concept: int = 3,
        min_success_for_probe: int = 5
    ) -> dict:
        """
        Run the full pipeline.

        Args:
            num_scenarios: Number of scenarios to generate
            target_model: Model to steer (for feature search and steering)
            max_features_per_concept: Max features to select per concept
            min_success_for_probe: Minimum successful results needed to create probe

        Returns:
            Dictionary with full pipeline results
        """
        # Create job record
        self.job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        jobs = load_pipeline_jobs()
        jobs.append({
            "id": self.job_id,
            "created_at": datetime.now().isoformat(),
            "config": {
                "num_scenarios": num_scenarios,
                "target_model": target_model,
                "high_quality_mode": self.high_quality_mode,
                "max_features_per_concept": max_features_per_concept
            },
            "status": "running",
            "progress": {},
            "results": None,
            "error": None
        })
        save_pipeline_jobs(jobs)

        try:
            return self._run_pipeline(
                num_scenarios,
                target_model,
                max_features_per_concept,
                min_success_for_probe
            )
        except Exception as e:
            # Update job with error
            jobs = load_pipeline_jobs()
            for job in jobs:
                if job["id"] == self.job_id:
                    job["status"] = "failed"
                    job["error"] = str(e)
                    break
            save_pipeline_jobs(jobs)
            raise

    def _run_pipeline(
        self,
        num_scenarios: int,
        target_model: str,
        max_features_per_concept: int,
        min_success_for_probe: int
    ) -> dict:
        """Internal pipeline execution."""

        results = {
            "job_id": self.job_id,
            "high_quality_mode": self.high_quality_mode,
            "templates_used": 0,
            "scenarios_generated": 0,
            "scenarios_approved": 0,
            "concepts_extracted": [],
            "features_selected": [],
            "steering_results": [],
            "evaluations": [],
            "successes": 0,
            "failures": 0,
            "review_queue_items": 0,
            "probe_created": False
        }

        # Step 1: Load vetted templates
        self._update_job_status("loading_templates")
        templates = load_vetted_templates()
        results["templates_used"] = len(templates)

        if not templates:
            raise ValueError("No vetted template scenarios found. Ensure scenarios exist in Fear/Survival or Corporate Loyalty categories.")

        # Step 2: Generate scenarios
        self._update_job_status("generating_scenarios", 0, num_scenarios)
        generated = self.scenario_agent.generate(templates, num_scenarios)
        results["scenarios_generated"] = len(generated)

        if not generated:
            raise ValueError("Scenario generation failed - no scenarios produced")

        # Step 3: Quality check
        self._update_job_status("quality_check", 0, len(generated))
        quality_result = self.quality_judge.evaluate(generated, templates)

        approved_scenarios = quality_result.get("approved", [])
        concepts = quality_result.get("extracted_concepts", [])

        results["scenarios_approved"] = len(approved_scenarios)
        results["concepts_extracted"] = concepts

        if not approved_scenarios:
            raise ValueError("No scenarios passed quality check")

        # Save generated scenarios
        existing_pipeline_scenarios = load_pipeline_scenarios()
        for scenario in approved_scenarios:
            scenario["pipeline_job_id"] = self.job_id
            scenario["created_at"] = datetime.now().isoformat()
        existing_pipeline_scenarios.extend(approved_scenarios)
        save_pipeline_scenarios(existing_pipeline_scenarios)

        # Step 4: Select features
        self._update_job_status("selecting_features", 0, len(concepts))
        concepts_used = concepts if concepts else ["deception", "dishonesty", "self-preservation"]
        if not concepts:
            results["concepts_extracted"] = concepts_used

        feature_result = self.feature_agent.search_and_select(
            concepts_used,
            target_model=target_model,
            max_per_concept=max_features_per_concept
        )
        selected_features = feature_result.get("selected_features", [])
        rejected_features = feature_result.get("rejected_features", [])
        feature_search_metadata = feature_result.get("search_metadata", {})
        overall_strategy = feature_result.get("overall_strategy", "")

        results["features_selected"] = selected_features
        results["layer_dedup_info"] = {
            "applied": feature_result.get("layer_deduplication_applied", False),
            "before": feature_result.get("features_before_dedup", len(selected_features)),
            "after": feature_result.get("features_after_dedup", len(selected_features))
        }

        if not selected_features:
            raise ValueError("No features selected for steering")

        # Load test strengths from config
        pipeline_config = load_pipeline_config()
        test_strengths = pipeline_config.get("steering_params", {}).get("test_strengths", [2, 5, 10])

        # Step 5: Run steering experiments (one feature at a time, multiple strengths)
        experiments = self._run_steering_experiments(
            scenarios=approved_scenarios,
            selected_features=selected_features,
            target_model=target_model,
            test_strengths=test_strengths
        )

        results["experiments"] = experiments
        results["test_strengths"] = test_strengths

        # Step 6: Evaluate results per feature, per strength
        self._update_job_status("evaluating", 0, len(experiments))
        feature_evaluations = []
        training_set = []
        review_items = 0
        total_successes = 0
        total_failures = 0

        for i, experiment in enumerate(experiments):
            self._update_job_status("evaluating", i + 1, len(experiments))

            feature = experiment["feature"]
            strength_results = experiment["strength_results"]

            feature_eval = {
                "feature": feature,
                "strength_evaluations": {},
                "best_strength": None,
                "best_score": -1,
                "recommendation": "flag_for_review"
            }

            for strength, scenario_results in strength_results.items():
                strength_evals = []
                strength_successes = 0
                strength_failures = 0

                for result in scenario_results:
                    if result.get("error"):
                        strength_evals.append({
                            "scenario_name": result["scenario_name"],
                            "evaluation": {
                                "classification": "failure",
                                "reason": f"Steering error: {result['error']}"
                            }
                        })
                        strength_failures += 1
                        continue

                    # Evaluate this specific strength result
                    evaluation = self.eval_judge.evaluate(
                        scenario=result["scenario"],
                        default_response=result["default"],
                        steered_response=result["steered"],
                        features_applied=[{
                            "layer": feature["layer"],
                            "index": feature["index"],
                            "strength": strength
                        }]
                    )

                    strength_evals.append({
                        "scenario_name": result["scenario_name"],
                        "evaluation": evaluation
                    })

                    classification = evaluation.get("classification", "inconclusive")
                    if classification == "success":
                        strength_successes += 1
                    elif classification == "failure":
                        strength_failures += 1

                feature_eval["strength_evaluations"][strength] = {
                    "evaluations": strength_evals,
                    "successes": strength_successes,
                    "failures": strength_failures,
                    "success_rate": strength_successes / len(scenario_results) if scenario_results else 0
                }

                # Track best strength for this feature
                success_rate = strength_successes / len(scenario_results) if scenario_results else 0
                if success_rate > feature_eval["best_score"]:
                    feature_eval["best_score"] = success_rate
                    feature_eval["best_strength"] = strength

            # Determine recommendation based on best strength performance
            if feature_eval["best_score"] >= 0.6:  # 60%+ success rate
                feature_eval["recommendation"] = "include_in_training"
                # Add best-strength results to training set
                if feature_eval["best_strength"] is not None:
                    best_results = strength_results.get(feature_eval["best_strength"], [])
                    for result in best_results:
                        if not result.get("error"):
                            training_set.append({
                                "scenario": result["scenario"],
                                "default": result["default"],
                                "steered": result["steered"],
                                "feature": feature,
                                "strength": feature_eval["best_strength"]
                            })
                            total_successes += 1
            elif feature_eval["best_score"] >= 0.3:  # 30-60% - needs review
                feature_eval["recommendation"] = "flag_for_review"
                # Add to review queue with best strength results
                if feature_eval["best_strength"] is not None:
                    best_results = strength_results.get(feature_eval["best_strength"], [])
                    for result in best_results:
                        if not result.get("error"):
                            add_to_review_queue(
                                scenario=result["scenario"],
                                default_response=result["default"],
                                steered_response=result["steered"],
                                features=[{
                                    "layer": feature["layer"],
                                    "index": feature["index"],
                                    "strength": feature_eval["best_strength"]
                                }],
                                evaluation=feature_eval,
                                job_id=self.job_id
                            )
                            review_items += 1
            else:
                feature_eval["recommendation"] = "reject"
                total_failures += 1

            feature_evaluations.append(feature_eval)

        results["feature_evaluations"] = feature_evaluations
        results["successes"] = total_successes
        results["failures"] = total_failures
        results["review_queue_items"] = review_items
        results["training_set_size"] = len(training_set)

        # Step 7: Create probe if enough successes
        self._update_job_status("finalizing")
        if len(training_set) >= min_success_for_probe:
            # TODO: Implement probe creation
            # This would call probe.py to create a MassMeanProbe
            results["probe_created"] = False  # Placeholder
            results["probe_note"] = f"Would create probe with {len(training_set)} examples"
        else:
            results["probe_note"] = f"Not enough successes ({len(training_set)}/{min_success_for_probe}) for probe creation"

        # Save detailed job results to separate file for transparency
        detailed_results = {
            "job_id": self.job_id,
            "created_at": datetime.now().isoformat(),
            "config": {
                "num_scenarios": num_scenarios,
                "target_model": target_model,
                "high_quality_mode": self.high_quality_mode,
                "max_features_per_concept": max_features_per_concept,
                "min_success_for_probe": min_success_for_probe,
                "agent_model": self.model,
                "test_strengths": test_strengths
            },
            "templates": templates,  # Full template scenarios used
            "generated_scenarios": generated,  # All scenarios before QA
            "quality_result": quality_result,  # Full QA output with rejections
            "approved_scenarios": approved_scenarios,  # Scenarios that passed QA
            "concepts_used": concepts_used,  # Concepts for feature search
            "feature_selection": {
                "selected_features": selected_features,
                "rejected_features": rejected_features,
                "search_metadata": feature_search_metadata,
                "overall_strategy": overall_strategy,
                "layer_deduplication": results.get("layer_dedup_info", {})
            },
            "experiments": experiments,  # Per-feature, per-strength steering results
            "feature_evaluations": feature_evaluations,  # Evaluation per feature with best strength
            "training_set": training_set,  # Successful examples
            "summary": {
                "templates_used": len(templates),
                "scenarios_generated": len(generated),
                "scenarios_approved": len(approved_scenarios),
                "concepts_count": len(concepts_used),
                "features_selected": len(selected_features),
                "features_rejected": len(rejected_features),
                "test_strengths": test_strengths,
                "total_steering_calls": len(selected_features) * len(test_strengths) * len(approved_scenarios),
                "successes": total_successes,
                "failures": total_failures,
                "review_items": review_items,
                "training_set_size": len(training_set)
            }
        }
        save_job_details(self.job_id, detailed_results)

        # Update job as completed
        jobs = load_pipeline_jobs()
        for job in jobs:
            if job["id"] == self.job_id:
                job["status"] = "completed"
                job["results"] = {
                    "templates_used": results["templates_used"],
                    "scenarios_generated": results["scenarios_generated"],
                    "scenarios_approved": results["scenarios_approved"],
                    "concepts": results["concepts_extracted"],
                    "features_count": len(results["features_selected"]),
                    "test_strengths": test_strengths,
                    "total_steering_calls": len(selected_features) * len(test_strengths) * len(approved_scenarios),
                    "successes": results["successes"],
                    "failures": results["failures"],
                    "review_items": results["review_queue_items"],
                    "training_set_size": len(training_set)
                }
                break
        save_pipeline_jobs(jobs)

        return results


def get_job_status(job_id: str) -> Optional[dict]:
    """Get the status of a pipeline job."""
    jobs = load_pipeline_jobs()
    for job in jobs:
        if job["id"] == job_id:
            return job
    return None


def cancel_job(job_id: str) -> bool:
    """Cancel a running pipeline job."""
    jobs = load_pipeline_jobs()
    for job in jobs:
        if job["id"] == job_id and job["status"] == "running":
            job["status"] = "cancelled"
            save_pipeline_jobs(jobs)
            return True
    return False


def list_jobs(limit: int = 20) -> list[dict]:
    """List recent pipeline jobs."""
    jobs = load_pipeline_jobs()
    return sorted(jobs, key=lambda x: x.get("created_at", ""), reverse=True)[:limit]


# Quick test function
def test_pipeline():
    """Test the pipeline with minimal configuration."""
    print("Testing pipeline...")

    orchestrator = PipelineOrchestrator(high_quality_mode=False)

    # Test with just 2 scenarios
    results = orchestrator.run(
        num_scenarios=2,
        target_model="llama3.1-8b-it",
        max_features_per_concept=2,
        min_success_for_probe=1
    )

    print("\n" + "=" * 50)
    print("Pipeline Results:")
    print(f"  Templates used: {results['templates_used']}")
    print(f"  Scenarios generated: {results['scenarios_generated']}")
    print(f"  Scenarios approved: {results['scenarios_approved']}")
    print(f"  Concepts: {results['concepts_extracted']}")
    print(f"  Features selected: {len(results['features_selected'])}")
    print(f"  Successes: {results['successes']}")
    print(f"  Failures: {results['failures']}")
    print(f"  Review queue items: {results['review_queue_items']}")
    print("=" * 50)

    return results


if __name__ == "__main__":
    test_pipeline()
