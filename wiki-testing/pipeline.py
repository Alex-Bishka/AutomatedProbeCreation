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
import re
from datetime import datetime
from typing import Optional
from pathlib import Path

from agents import (
    ScenarioCreatorAgent,
    ScenarioQualityJudge,
    FeatureSelectorAgent,
    SteeringDirectionAgent,
    FeatureMixAgent,
    EvaluationJudgeAgent,
    DEFAULT_MODEL,
    HIGH_QUALITY_MODEL,
    VETTED_CATEGORIES,
    get_agent,
    load_pipeline_config
)
from neuronpedia import steering_chat
from logger import logger

# Paths
FRONTEND_DIR = Path(__file__).parent / "frontend"
SCENARIOS_FILE = FRONTEND_DIR / "scenarios.json"
PIPELINE_SCENARIOS_FILE = FRONTEND_DIR / "pipeline_scenarios.json"
PIPELINE_JOBS_FILE = FRONTEND_DIR / "pipeline_jobs.json"
REVIEW_QUEUE_FILE = FRONTEND_DIR / "review_queue.json"
SCENARIO_BANK_FILE = FRONTEND_DIR / "scenario_bank.json"
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


def load_scenario_bank() -> dict:
    """Load the scenario bank."""
    if SCENARIO_BANK_FILE.exists():
        with open(SCENARIO_BANK_FILE, 'r') as f:
            return json.load(f)
    return {"scenarios": [], "tags": []}


def load_vetted_templates() -> list[dict]:
    """
    Load template scenarios for generation.

    Loads from vetted categories (Fear/Survival + Corporate Loyalty),
    and optionally supplements with enabled scenarios from the scenario bank.
    """
    # Load from vetted categories
    scenarios = load_scenarios()
    templates = [s for s in scenarios if s.get("category") in VETTED_CATEGORIES]

    # Check if scenario bank is enabled
    config = load_pipeline_config()
    scenario_sources = config.get("scenario_sources", {})

    if scenario_sources.get("use_scenario_bank", False):
        bank = load_scenario_bank()
        bank_scenarios = bank.get("scenarios", [])

        # Filter by enabled and optionally by tags
        filter_tags = scenario_sources.get("scenario_bank_tags", [])

        for scenario in bank_scenarios:
            if not scenario.get("enabled", True):
                continue

            # If tags are specified, scenario must have at least one matching tag
            if filter_tags:
                scenario_tags = scenario.get("tags", [])
                if not any(tag in scenario_tags for tag in filter_tags):
                    continue

            # Add if not already present (by name)
            if not any(t.get("name") == scenario.get("name") for t in templates):
                templates.append(scenario)

        logger.info(f"Loaded {len(templates)} templates ({len(templates) - len([s for s in scenarios if s.get('category') in VETTED_CATEGORIES])} from scenario bank)")

    return templates


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
        self.direction_agent = SteeringDirectionAgent(self.model)
        self.mix_agent = FeatureMixAgent(self.model)
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

    def _filter_strengths_by_direction(
        self,
        all_strengths: list[int],
        direction: str
    ) -> list[int]:
        """
        Filter strengths to only include values matching the specified direction.

        Args:
            all_strengths: Full list of test strengths (e.g., [-10, -5, -2, 2, 5, 10])
            direction: "positive" or "negative"

        Returns:
            Filtered list containing only positive or negative values
        """
        if direction == "positive":
            return [s for s in all_strengths if s > 0]
        elif direction == "negative":
            return [s for s in all_strengths if s < 0]
        else:
            # Unknown direction, return all (fallback)
            return all_strengths

    def _run_steering_experiments(
        self,
        scenarios: list[dict],
        selected_features: list[dict],
        target_model: str,
        test_strengths: list[int],
        feature_directions: dict = None
    ) -> list[dict]:
        """
        Test each feature individually at multiple strength levels.

        Args:
            scenarios: List of approved scenarios to test
            selected_features: Features to test (one per layer after dedup)
            target_model: Model to steer
            test_strengths: List of strength values to test (e.g., [-10, -5, -2, 2, 5, 10])
            feature_directions: Optional dict mapping (layer, index) -> {"direction": "positive"|"negative"}
                              If provided, strengths are filtered per-feature based on direction.

        Returns:
            List of experiment results, one per feature:
            [
                {
                    "feature": {...},
                    "direction": "positive"|"negative",
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

        # Calculate total experiments accounting for direction filtering
        total_experiments = 0
        for feature in selected_features:
            key = (feature.get("layer", ""), int(feature.get("index", 0)))
            if feature_directions and key in feature_directions:
                direction = feature_directions[key].get("direction", "positive")
                filtered = self._filter_strengths_by_direction(test_strengths, direction)
                total_experiments += len(filtered) * len(scenarios)
            else:
                total_experiments += len(test_strengths) * len(scenarios)

        current_experiment = 0

        # Cache default responses per scenario (same across all features)
        default_responses_cache = {}

        for feature in selected_features:
            key = (feature.get("layer", ""), int(feature.get("index", 0)))

            # Determine direction and filter strengths for this feature
            if feature_directions and key in feature_directions:
                direction_info = feature_directions[key]
                direction = direction_info.get("direction", "positive")
                direction_reasoning = direction_info.get("reasoning", "")
            else:
                direction = "positive"  # Default to positive
                direction_reasoning = "Default: no direction specified"

            feature_strengths = self._filter_strengths_by_direction(test_strengths, direction)

            feature_experiment = {
                "feature": {
                    "layer": feature.get("layer", ""),
                    "index": feature.get("index", 0),
                    "description": feature.get("description", ""),
                    "cosine_similarity": feature.get("cosine_similarity", 0),
                    "concept": feature.get("concept", "")
                },
                "direction": direction,
                "direction_reasoning": direction_reasoning,
                "strength_results": {}
            }

            for strength in feature_strengths:
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

    def _coherence_metrics(self, text: str) -> dict:
        """Compute simple coherence metrics for a text response."""
        if not text:
            return {
                "score": 0.0,
                "word_count": 0,
                "non_ascii_ratio": 1.0,
                "alnum_ratio": 0.0,
                "printable_ratio": 0.0
            }

        total = len(text)
        ascii_count = sum(1 for c in text if ord(c) < 128)
        printable_count = sum(1 for c in text if c.isprintable())
        alnum_space_count = sum(1 for c in text if c.isalnum() or c.isspace())
        words = re.findall(r"[A-Za-z]{2,}", text)
        word_count = len(words)

        non_ascii_ratio = 1.0 - (ascii_count / total)
        printable_ratio = printable_count / total
        alnum_ratio = alnum_space_count / total

        score = (0.5 * printable_ratio) + (0.3 * alnum_ratio) + (0.2 * min(word_count / 20, 1.0))
        score -= non_ascii_ratio * 0.5
        score = max(0.0, min(1.0, score))

        return {
            "score": score,
            "word_count": word_count,
            "non_ascii_ratio": non_ascii_ratio,
            "alnum_ratio": alnum_ratio,
            "printable_ratio": printable_ratio
        }

    def _is_coherent(self, metrics: dict, params: dict) -> bool:
        """Determine if a response is coherent based on thresholds."""
        min_score = params.get("min_coherence_score", 0.55)
        min_words = params.get("min_word_count", 5)
        max_non_ascii = params.get("max_non_ascii_ratio", 0.2)
        min_alnum_ratio = params.get("min_alnum_ratio", 0.25)

        return (
            metrics.get("score", 0) >= min_score
            and metrics.get("word_count", 0) >= min_words
            and metrics.get("non_ascii_ratio", 1.0) <= max_non_ascii
            and metrics.get("alnum_ratio", 0.0) >= min_alnum_ratio
        )

    def _build_mix_candidates(
        self,
        feature_evaluations: list[dict],
        experiments: list[dict],
        coherence_params: dict
    ) -> list[dict]:
        """Build candidate list for mix selection from per-feature evaluations."""
        experiment_map = {}
        for experiment in experiments:
            feature = experiment.get("feature", {})
            key = (feature.get("layer", ""), int(feature.get("index", 0)))
            experiment_map[key] = experiment

        candidates = []
        for feat_eval in feature_evaluations:
            feature = feat_eval.get("feature", {})
            best_strength = feat_eval.get("best_strength")
            if best_strength is None:
                continue

            avg_coherence = 0.0
            incoherent_rate = 0.0
            experiment = experiment_map.get((feature.get("layer", ""), int(feature.get("index", 0))))
            if experiment:
                strength_results = experiment.get("strength_results", {})
                results = strength_results.get(best_strength)
                if results is None:
                    results = strength_results.get(str(best_strength))

                if results:
                    scores = []
                    incoherent = 0
                    for result in results:
                        metrics = self._coherence_metrics(result.get("steered", ""))
                        scores.append(metrics["score"])
                        if not self._is_coherent(metrics, coherence_params):
                            incoherent += 1
                    avg_coherence = sum(scores) / len(scores) if scores else 0.0
                    incoherent_rate = incoherent / len(scores) if scores else 0.0

            if avg_coherence and avg_coherence < coherence_params.get("min_coherence_score", 0.55):
                continue
            if incoherent_rate > coherence_params.get("max_incoherent_rate_per_feature", 0.4):
                continue

            candidates.append({
                "layer": feature.get("layer", ""),
                "index": feature.get("index", 0),
                "description": feature.get("description", ""),
                "concept": feature.get("concept", ""),
                "best_strength": best_strength,
                "best_score": feat_eval.get("best_score", 0),
                "recommendation": feat_eval.get("recommendation", ""),
                "coherence_score": avg_coherence,
                "incoherent_rate": incoherent_rate
            })
        return candidates

    def _normalize_mixed_features(self, mix_result: dict, candidates: list[dict]) -> list[dict]:
        """Normalize mixed features and fill missing strengths from candidates."""
        if not mix_result:
            return []

        raw_features = mix_result.get("mixed_features", [])
        if not isinstance(raw_features, list):
            return []

        candidate_map = {}
        for candidate in candidates:
            key = (candidate.get("layer", ""), int(candidate.get("index", 0)))
            candidate_map[key] = candidate

        normalized = []
        for feature in raw_features:
            layer = feature.get("layer", "")
            index = feature.get("index", None)
            if not layer or index is None:
                continue
            try:
                index = int(index)
            except (TypeError, ValueError):
                continue

            candidate = candidate_map.get((layer, index), {})
            strength = feature.get("strength")
            if strength is None:
                strength = candidate.get("best_strength")
            if strength is None:
                continue

            normalized.append({
                "layer": layer,
                "index": index,
                "strength": strength,
                "concept": feature.get("concept", candidate.get("concept", "")),
                "description": candidate.get("description", ""),
                "reasoning": feature.get("reasoning", "")
            })

        return normalized

    def _fallback_mixed_features(self, candidates: list[dict], max_features: int) -> list[dict]:
        """Fallback mix selection when the mix agent fails."""
        def score(candidate: dict) -> tuple:
            recommendation = candidate.get("recommendation", "")
            rec_score = 1 if recommendation == "include_in_training" else 0
            return (rec_score, candidate.get("best_score", 0))

        ranked = sorted(candidates, key=score, reverse=True)
        mixed = []
        for candidate in ranked[:max_features]:
            if candidate.get("best_strength") is None:
                continue
            mixed.append({
                "layer": candidate.get("layer", ""),
                "index": int(candidate.get("index", 0)),
                "strength": candidate.get("best_strength"),
                "concept": candidate.get("concept", ""),
                "description": candidate.get("description", ""),
                "reasoning": "Fallback mix based on single-feature success rate."
            })
        return mixed

    def _run_mixed_steering(
        self,
        scenarios: list[dict],
        mixed_features: list[dict],
        target_model: str,
        coherence_params: dict
    ) -> list[dict]:
        """Run steering with a combined feature mix across scenarios."""
        results = []
        steering_config = []
        for feature in mixed_features:
            steering_config.append({
                "modelId": target_model,
                "layer": feature.get("layer", ""),
                "index": feature.get("index", 0),
                "strength": feature.get("strength", 0)
            })

        for scenario in scenarios:
            scenario_name = scenario.get("name", "Unknown")
            messages = scenario.get("messages", [])
            try:
                default_resp, steered_resp = steering_chat(
                    messages,
                    steering_config,
                    model=target_model
                )
                default_text = extract_response(default_resp)
                steered_text = extract_response(steered_resp)

                evaluation = self.eval_judge.evaluate(
                    scenario=scenario,
                    default_response=default_text,
                    steered_response=steered_text,
                    features_applied=steering_config
                )

                coherence = self._coherence_metrics(steered_text)
                is_coherent = self._is_coherent(coherence, coherence_params)
                if not is_coherent:
                    evaluation = evaluation or {}
                    evaluation["classification"] = "failure"
                    evaluation["recommendation"] = "exclude"
                    evaluation["review_reason"] = "Incoherent output (automatic filter)"
                    evaluation["coherence_override"] = True

                results.append({
                    "scenario_name": scenario_name,
                    "scenario": scenario,
                    "default": default_text,
                    "steered": steered_text,
                    "evaluation": evaluation,
                    "coherence": coherence,
                    "error": None
                })
            except Exception as e:
                results.append({
                    "scenario_name": scenario_name,
                    "scenario": scenario,
                    "default": "",
                    "steered": "",
                    "evaluation": None,
                    "error": str(e)
                })

        return results

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

        logger.info("="*60)
        logger.info(f"Starting pipeline job: {self.job_id}")
        logger.info(f"Configuration:")
        logger.info(f"  - num_scenarios: {num_scenarios}")
        logger.info(f"  - target_model: {target_model}")
        logger.info(f"  - high_quality_mode: {self.high_quality_mode}")
        logger.info(f"  - max_features_per_concept: {max_features_per_concept}")
        logger.info(f"  - min_success_for_probe: {min_success_for_probe}")
        logger.info("="*60)

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

        # Load pipeline configuration once at the start
        pipeline_config = load_pipeline_config()

        # Step 1: Load vetted templates
        self._update_job_status("loading_templates")
        templates = load_vetted_templates()
        results["templates_used"] = len(templates)

        logger.info(f"Step 1: Loaded {len(templates)} vetted template scenarios")

        if not templates:
            raise ValueError("No vetted template scenarios found. Ensure scenarios exist in Fear/Survival or Corporate Loyalty categories.")

        # Step 2: Generate scenarios
        self._update_job_status("generating_scenarios", 0, num_scenarios)
        generated = self.scenario_agent.generate(templates, num_scenarios)
        results["scenarios_generated"] = len(generated)

        logger.info(f"Step 2: Generated {len(generated)} scenarios")

        if not generated:
            raise ValueError("Scenario generation failed - no scenarios produced")

        # Step 3: Quality check with retry for concept extraction
        self._update_job_status("quality_check", 0, len(generated))

        max_concept_retries = 3
        approved_scenarios = []
        concepts = []

        for attempt in range(1, max_concept_retries + 1):
            logger.info(f"Quality check attempt {attempt}/{max_concept_retries}")

            quality_result = self.quality_judge.evaluate(generated, templates)
            approved_scenarios = quality_result.get("approved", [])
            concepts = quality_result.get("extracted_concepts", [])

            logger.info(f"Attempt {attempt}: {len(approved_scenarios)} scenarios approved, {len(concepts)} concepts extracted")

            if concepts:
                logger.info(f"Concepts extracted: {concepts}")
                break
            else:
                logger.warning(f"Attempt {attempt}: No concepts extracted from quality judge")
                if attempt < max_concept_retries:
                    logger.info(f"Retrying quality evaluation (attempt {attempt + 1}/{max_concept_retries})")

        # After all retries, check if we have concepts
        if not concepts:
            error_msg = f"Failed to extract concepts after {max_concept_retries} attempts. Quality judge evaluation unsuccessful."
            logger.error(error_msg)
            raise ValueError(error_msg)

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
        # Load topk from config (number of search results to analyze per concept)
        topk_search_results = pipeline_config.get("pipeline_defaults", {}).get("topk_search_results", 20)

        logger.info(f"Step 4: Starting feature selection for {len(concepts)} concepts")
        logger.info(f"  - Concepts: {concepts}")
        logger.info(f"  - Top-K search results per concept: {topk_search_results}")
        self._update_job_status("selecting_features", 0, len(concepts))

        feature_result = self.feature_agent.search_and_select(
            concepts,
            target_model=target_model,
            max_per_concept=max_features_per_concept,
            topk=topk_search_results
        )
        selected_features = feature_result.get("selected_features", [])
        rejected_features = feature_result.get("rejected_features", [])
        feature_search_metadata = feature_result.get("search_metadata", {})
        overall_strategy = feature_result.get("overall_strategy", "")

        # Log feature selection results
        logger.info(f"Feature selection complete:")
        logger.info(f"  - Selected: {len(selected_features)} features")
        logger.info(f"  - Rejected: {len(rejected_features)} features")

        # Log per-concept breakdown
        for concept, meta in feature_search_metadata.items():
            selected_count = len([f for f in selected_features if f.get("concept") == concept])
            logger.info(f"  - Concept '{concept}': {selected_count} features selected from {meta.get('search_results_count', 0)} search results")

        if selected_features:
            logger.info(f"Selected features summary:")
            for feat in selected_features:
                logger.info(f"  - Layer {feat.get('layer')}, Index {feat.get('index')}: {feat.get('description', '')[:80]}...")

        results["features_selected"] = selected_features
        results["features_rejected_count"] = len(rejected_features)
        results["layer_dedup_info"] = {
            "applied": feature_result.get("layer_deduplication_applied", False),
            "before": feature_result.get("features_before_dedup", len(selected_features)),
            "after": feature_result.get("features_after_dedup", len(selected_features))
        }

        if not selected_features:
            logger.error("No features selected for steering")
            raise ValueError("No features selected for steering")

        # Load test strengths from config
        test_strengths = pipeline_config.get("steering_params", {}).get("test_strengths", [2, 5, 10])

        # Step 4.5: Determine steering direction for each feature
        logger.info(f"Step 4.5: Determining steering direction for each feature")
        self._update_job_status("determining_directions", 0, len(selected_features))

        feature_directions = self.direction_agent.batch_decide(
            scenarios=approved_scenarios,
            features=selected_features,
            target_behavior="deception"
        )

        # Log direction decisions
        positive_count = sum(1 for d in feature_directions.values() if d.get("direction") == "positive")
        negative_count = sum(1 for d in feature_directions.values() if d.get("direction") == "negative")
        logger.info(f"Direction decisions: {positive_count} positive, {negative_count} negative")

        for (layer, index), direction_info in feature_directions.items():
            logger.info(f"  - Layer {layer}, Index {index}: {direction_info.get('direction')} ({direction_info.get('reasoning', '')[:60]}...)")

        results["feature_directions"] = {
            f"{layer}:{index}": direction_info
            for (layer, index), direction_info in feature_directions.items()
        }

        # Calculate actual experiment count (accounting for direction filtering)
        positive_strengths = [s for s in test_strengths if s > 0]
        negative_strengths = [s for s in test_strengths if s < 0]
        actual_experiments = 0
        for feature in selected_features:
            key = (feature.get("layer", ""), int(feature.get("index", 0)))
            direction = feature_directions.get(key, {}).get("direction", "positive")
            if direction == "positive":
                actual_experiments += len(positive_strengths) * len(approved_scenarios)
            else:
                actual_experiments += len(negative_strengths) * len(approved_scenarios)

        logger.info(f"Step 5: Starting steering experiments")
        logger.info(f"  - Features to test: {len(selected_features)}")
        logger.info(f"  - Test strengths (all): {test_strengths}")
        logger.info(f"  - Positive strengths: {positive_strengths}")
        logger.info(f"  - Negative strengths: {negative_strengths}")
        logger.info(f"  - Scenarios: {len(approved_scenarios)}")
        logger.info(f"  - Total experiments (direction-filtered): {actual_experiments}")

        # Step 5: Run steering experiments (one feature at a time, direction-filtered strengths)
        experiments = self._run_steering_experiments(
            scenarios=approved_scenarios,
            selected_features=selected_features,
            target_model=target_model,
            test_strengths=test_strengths,
            feature_directions=feature_directions
        )

        results["experiments"] = experiments
        results["test_strengths"] = test_strengths

        # Step 6: Evaluate results per feature, per strength
        logger.info(f"Step 6: Evaluating steering results")
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

        # Step 6.5: Build and test a mixed-feature steering set (iterative)
        logger.info("Step 6.5: Building mixed-feature steering set")
        self._update_job_status("mixing_features")

        default_mixing_params = {
            "max_attempts": 3,
            "min_coherence_score": 0.55,
            "min_word_count": 5,
            "max_non_ascii_ratio": 0.2,
            "min_alnum_ratio": 0.25,
            "max_incoherent_fraction": 0.3,
            "max_incoherent_rate_per_feature": 0.4,
            "min_approved": 1
        }
        mixing_params = {**default_mixing_params, **pipeline_config.get("mixing_params", {})}

        max_combined_features = pipeline_config.get("steering_params", {}).get("max_combined_features", 3)
        mix_candidates = self._build_mix_candidates(feature_evaluations, experiments, mixing_params)

        mix_result = None
        mixed_features = []
        mixed_results = []
        mix_attempts = []
        feedback = ""

        for attempt in range(1, mixing_params["max_attempts"] + 1):
            if not mix_candidates:
                logger.warning("No viable mix candidates after coherence filtering")
                break

            mix_result = self.mix_agent.create_mix(
                mix_candidates,
                max_features=max_combined_features,
                feedback=feedback
            )
            mixed_features = self._normalize_mixed_features(mix_result or {}, mix_candidates)

            if not mixed_features and mix_candidates:
                mixed_features = self._fallback_mixed_features(mix_candidates, max_combined_features)
                mix_result = mix_result or {}
                mix_result.setdefault("rationale", "Fallback mix based on single-feature success rates.")

            if not mixed_features:
                logger.warning("Mix agent produced no usable features")
                break

            logger.info(f"Running mixed-feature steering (attempt {attempt}) with {len(mixed_features)} features")
            mixed_results = self._run_mixed_steering(
                approved_scenarios,
                mixed_features,
                target_model,
                mixing_params
            )

            total = len(mixed_results)
            incoherent = sum(1 for r in mixed_results if r.get("evaluation", {}).get("coherence_override"))
            approved = sum(1 for r in mixed_results if r.get("evaluation", {}).get("recommendation") == "include_in_training")
            incoherent_fraction = (incoherent / total) if total else 1.0

            mix_attempts.append({
                "attempt": attempt,
                "features": mixed_features,
                "summary": {
                    "total": total,
                    "approved": approved,
                    "incoherent": incoherent,
                    "incoherent_fraction": incoherent_fraction
                },
                "feedback": feedback
            })

            if incoherent_fraction > mixing_params["max_incoherent_fraction"]:
                feedback = (
                    f"Outputs were incoherent in {incoherent}/{total} cases "
                    f"({incoherent_fraction:.2f}). Select fewer features with higher coherence_score "
                    "and avoid conflicting directions."
                )
                continue

            if approved < mixing_params["min_approved"]:
                feedback = (
                    f"Only {approved}/{total} were approved. Choose features with higher best_score "
                    "and stronger single-feature performance."
                )
                continue

            break

        if not mixed_features:
            logger.warning("No mixed-feature steering set created")

        mix_summary = {
            "total": len(mixed_results),
            "approved": sum(
                1 for r in mixed_results
                if r.get("evaluation", {}).get("recommendation") == "include_in_training"
            )
        }
        mix_summary["status"] = "success" if mix_summary["approved"] >= mixing_params["min_approved"] else "failure"

        sample_outputs = []
        for result in mixed_results:
            if result.get("steered"):
                sample_outputs.append(result["steered"][:400])
            if len(sample_outputs) >= 3:
                break

        outcome_explanation = ""
        if mixed_features and mixed_results:
            outcome_explanation = self.mix_agent.explain_outcome(
                summary=mix_summary,
                mixed_features=mixed_features,
                sample_outputs=sample_outputs
            )

        results["mixed_steering"] = {
            "features": mixed_features,
            "rationale": (mix_result or {}).get("rationale", ""),
            "expected_effect": (mix_result or {}).get("expected_effect", ""),
            "risk_notes": (mix_result or {}).get("risk_notes", ""),
            "results": mixed_results,
            "attempts": mix_attempts,
            "summary": mix_summary,
            "outcome_explanation": outcome_explanation
        }

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
            "concepts": concepts,  # Concepts for feature search
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
            "mixed_steering": results.get("mixed_steering", {}),
            "summary": {
                "templates_used": len(templates),
                "scenarios_generated": len(generated),
                "scenarios_approved": len(approved_scenarios),
                "concepts_count": len(concepts),
                "features_selected": len(selected_features),
                "features_rejected": len(rejected_features),
                "test_strengths": test_strengths,
                "total_steering_calls": len(selected_features) * len(test_strengths) * len(approved_scenarios),
                "successes": total_successes,
                "failures": total_failures,
                "review_items": review_items,
                "training_set_size": len(training_set),
                "mixed_total": results.get("mixed_steering", {}).get("summary", {}).get("total", 0),
                "mixed_approved": results.get("mixed_steering", {}).get("summary", {}).get("approved", 0)
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

        logger.info("="*60)
        logger.info(f"Pipeline job {self.job_id} completed successfully")
        logger.info(f"Results summary:")
        logger.info(f"  - Scenarios generated: {results['scenarios_generated']}")
        logger.info(f"  - Scenarios approved: {results['scenarios_approved']}")
        logger.info(f"  - Concepts extracted: {len(results['concepts_extracted'])}")
        logger.info(f"  - Features selected: {len(results['features_selected'])}")
        logger.info(f"  - Successes: {results['successes']}")
        logger.info(f"  - Failures: {results['failures']}")
        logger.info(f"  - Review items: {results['review_queue_items']}")
        logger.info("="*60)

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
