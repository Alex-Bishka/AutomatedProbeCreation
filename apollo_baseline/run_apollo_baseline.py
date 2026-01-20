from __future__ import annotations

from pathlib import Path
from functools import cache
import os
import sys
from typing import Any, Literal, Optional, cast

import fire
import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[1]
DECEPTION_DET_ROOT = REPO_ROOT / "deception-detection"
if str(DECEPTION_DET_ROOT) not in sys.path:
    sys.path.insert(0, str(DECEPTION_DET_ROOT))

import deception_detection.models as dd_models
from deception_detection.experiment import Experiment, ExperimentConfig
from deception_detection.log import logger
from deception_detection.models import ModelName, get_num_layers_for_model
from deception_detection.postprocess import ExperimentPostprocessor
from deception_detection.prompt_scorer import PromptScorer, PromptScorerMethod
from deception_detection.repository import DatasetRepository
from deception_detection.results.detection_masks import save_detection_masks
from deception_detection.rollouts import RolloutGenerator
from deception_detection.utils import REPO_ROOT as DD_REPO_ROOT

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "apollo_llama3_1_8b_all_layers.yaml"


# -----------------------------------------------------------------------------
# GPU max_memory helpers
# -----------------------------------------------------------------------------


def _get_max_memory_map() -> dict[int, str] | None:
    if not torch.cuda.is_available():
        return None

    cap_env = os.getenv("APOLLO_MAX_GPU_MEM_GB")
    cap_gb = float(cap_env) if cap_env else None
    max_memory: dict[int, str] = {}
    for device_idx in range(torch.cuda.device_count()):
        total_gb = torch.cuda.get_device_properties(device_idx).total_memory / (1024**3)
        use_gb = min(total_gb * 0.98, cap_gb) if cap_gb is not None else total_gb * 0.98
        max_memory[device_idx] = f"{int(use_gb)}GiB"
    return max_memory


# -----------------------------------------------------------------------------
# Hugging Face cache + gated model access
# -----------------------------------------------------------------------------


def _resolve_hf_cache_dir() -> Path:
    """
    Resolve a writable cache directory for Hugging Face downloads.

    Priority:
      1) HF_HOME (recommended)
      2) HF_HUB_CACHE parent
      3) TRANSFORMERS_CACHE parent
      4) $SCRATCH/hf_cache/$USER if SCRATCH set
      5) ~/.cache/huggingface
    """
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home)

    hf_hub_cache = os.environ.get("HF_HUB_CACHE")
    if hf_hub_cache:
        return Path(hf_hub_cache).parent

    tf_cache = os.environ.get("TRANSFORMERS_CACHE")
    if tf_cache:
        return Path(tf_cache).parent

    scratch = os.environ.get("SCRATCH")
    user = os.environ.get("USER", "user")
    if scratch:
        return Path(scratch) / "hf_cache" / user

    return Path.home() / ".cache" / "huggingface"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _patch_llama_model_loading() -> None:
    """
    Patch dd_models.get_llama3_model_and_tokenizer so it never writes to /data and always
    uses a writable cache_dir + respects HF_TOKEN for gated models.
    """

    @cache
    def _patched_get_llama3_model_and_tokenizer(
        model_name: ModelName, omit_model: bool = False
    ):
        tokenizer = dd_models.get_llama3_tokenizer()
        if omit_model:
            return None, tokenizer

        model_path = {
            ModelName.LLAMA_1B: "meta-llama/Llama-3.2-1B-Instruct",
            ModelName.LLAMA_3B: "meta-llama/Llama-3.2-3B-Instruct",
            ModelName.LLAMA_70B_3_3: "meta-llama/Llama-3.3-70B-Instruct",
            ModelName.LLAMA_70B: "meta-llama/Meta-Llama-3.1-70B-Instruct",
            ModelName.LLAMA_8B: "meta-llama/Meta-Llama-3.1-8B-Instruct",
        }[model_name]

        cache_root = _resolve_hf_cache_dir()
        models_directory = cache_root / "transformers"
        _ensure_dir(models_directory)

        token: Optional[str] = os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGINGFACE_HUB_TOKEN"
        )

        max_memory = _get_max_memory_map()
        if max_memory:
            logger.info(f"Using max_memory map for {len(max_memory)} GPUs: {max_memory}")
        logger.info(f"Loading {model_path} with cache_dir={models_directory}")

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                max_memory=max_memory,
                cache_dir=str(models_directory),
                local_files_only=False,
                token=token,
            )
        except TypeError:
            # Older transformers: use_auth_token instead of token
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                max_memory=max_memory,
                cache_dir=str(models_directory),
                local_files_only=False,
                use_auth_token=token,
            )

        model.eval()
        return model, tokenizer

    dd_models.get_llama3_model_and_tokenizer = _patched_get_llama3_model_and_tokenizer


_patch_llama_model_loading()


# -----------------------------------------------------------------------------
# Postprocess crash fixes: ai_liar + roleplaying plots
# -----------------------------------------------------------------------------


def _patch_postprocess_plotting() -> None:
    """
    Prevent postprocess from crashing on missing columns in dataset-specific plots.

    Handles:
      - ai_liar_violin: missing 'expected_label' column.
      - roleplaying_violin: missing 'model_graded_deception' column.
    """
    try:
        from deception_detection.results import plot as plot_mod  # type: ignore
    except Exception as e:
        logger.warning("Could not import deception_detection.results.plot for patching: %s", e)
        return

    if getattr(plot_mod, "_apollo_plot_patch_applied", False):
        return

    # Patch ai_liar_violin
    if hasattr(plot_mod, "ai_liar_violin"):
        orig_ai_liar = plot_mod.ai_liar_violin

        def safe_ai_liar_violin(*args: Any, **kwargs: Any):
            try:
                return orig_ai_liar(*args, **kwargs)
            except KeyError as e:
                if str(e).strip().strip("'") == "expected_label":
                    logger.warning(
                        "Skipping ai_liar violin plot: dataframe missing expected_label column."
                    )
                    return None
                raise

        plot_mod.ai_liar_violin = safe_ai_liar_violin  # type: ignore[assignment]

    # Patch roleplaying_violin
    if hasattr(plot_mod, "roleplaying_violin"):
        orig_roleplaying = plot_mod.roleplaying_violin

        def safe_roleplaying_violin(*args: Any, **kwargs: Any):
            try:
                return orig_roleplaying(*args, **kwargs)
            except KeyError as e:
                if str(e).strip().strip("'") == "model_graded_deception":
                    logger.warning(
                        "Skipping roleplaying violin plot: dataframe missing "
                        "model_graded_deception column."
                    )
                    return None
                raise

        plot_mod.roleplaying_violin = safe_roleplaying_violin  # type: ignore[assignment]

    # Patch violin_plot (can fail on mismatched array lengths in dataframe construction)
    if hasattr(plot_mod, "violin_plot"):
        orig_violin_plot = plot_mod.violin_plot

        def safe_violin_plot(*args: Any, **kwargs: Any):
            try:
                return orig_violin_plot(*args, **kwargs)
            except ValueError as e:
                if "All arrays must be of the same length" in str(e):
                    logger.warning("Skipping violin plot: %s", e)
                    return None
                raise

        plot_mod.violin_plot = safe_violin_plot  # type: ignore[assignment]

    plot_mod._apollo_plot_patch_applied = True


# -----------------------------------------------------------------------------
# In-memory diagnostics on the results DataFrame
# -----------------------------------------------------------------------------


def _write_results_diagnostics_in_memory(
    folder: Path,
    postprocessor: ExperimentPostprocessor,
) -> None:
    """
    Use the in-memory results DataFrame to compute simple diagnostics, without
    re-parsing any JSON.

    Adds, per dataset:
      - diagnostic_impacted: bool
      - diagnostic_issues: comma-separated reasons (e.g. no_honest, no_deceptive, auroc_nan)
      - auroc_marked: AUROC with '*' appended if impacted

    Writes:
      - <folder>/results_with_diagnostics_in_memory.csv
    And logs a compact table.
    """
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        logger.warning("Pandas not available; skipping in-memory diagnostics: %s", e)
        return

    try:
        df = postprocessor.get_results_dataframe()  # type: ignore[attr-defined]
    except Exception as e:
        logger.warning("Could not obtain results DataFrame from postprocessor: %s", e)
        return

    # Work on a copy so we don't mutate whatever postprocess uses internally.
    df = df.copy()

    # Ensure we have a 'dataset' column (dataset names are often the index).
    if "dataset" not in df.columns:
        df.insert(0, "dataset", df.index.astype(str))

    # Heuristic diagnostics based only on the summary stats we have.
    issues_col: list[str] = []
    impacted_col: list[bool] = []
    auroc_marked_col: list[str] = []

    for _, row in df.iterrows():
        issues: list[str] = []

        # Basic counts: if we have zero honest or zero deceptive, AUROC is structurally dodgy.
        n_honest = int(row.get("n_honest", 0) or 0)
        n_deceptive = int(row.get("n_deceptive", 0) or 0)
        auroc = float(row.get("auroc", float("nan")))

        if n_honest == 0:
            issues.append("no_honest")
        if n_deceptive == 0:
            issues.append("no_deceptive")
        if not np.isfinite(auroc):
            issues.append("auroc_nan_or_inf")

        impacted = len(issues) > 0
        impacted_col.append(impacted)
        issues_col.append(",".join(issues))

        if not np.isfinite(auroc):
            au_str = "nan"
        else:
            au_str = f"{auroc:.3f}"

        if impacted and au_str != "nan":
            auroc_marked_col.append(au_str + "*")
        else:
            auroc_marked_col.append(au_str)

    df["diagnostic_issues"] = issues_col
    df["diagnostic_impacted"] = impacted_col
    df["auroc_marked"] = auroc_marked_col

    outfile = folder / "results_with_diagnostics_in_memory.csv"
    try:
        df.to_csv(outfile, index=False)
        view_cols = [
            "dataset",
            "n_honest",
            "n_deceptive",
            "auroc_marked",
            "diagnostic_impacted",
            "diagnostic_issues",
        ]
        # Filter to view columns that actually exist
        view_cols = [c for c in view_cols if c in df.columns]
        logger.info(
            "In-memory diagnostics summary (AUROC values with * are flagged as impacted):\n%s",
            df[view_cols].to_string(index=False),
        )
        logger.info("Saved in-memory diagnostics to %s", outfile)
    except Exception as e:
        logger.warning("Failed to write in-memory diagnostics CSV: %s", e)


# -----------------------------------------------------------------------------
# Config helpers
# -----------------------------------------------------------------------------


def _load_config(config_file: Path) -> dict[str, Any]:
    with open(config_file) as file:
        return cast(dict[str, Any], yaml.safe_load(file))


def _maybe_set_all_layers(config_dict: dict[str, Any]) -> None:
    detect_layers = config_dict.get("detect_layers")
    if detect_layers:
        return

    model_name = ModelName(config_dict["model_name"])
    n_layers = get_num_layers_for_model(model_name)
    config_dict["detect_layers"] = list(range(n_layers))


def _ensure_on_policy_rollouts(config_dict: dict[str, Any]) -> None:
    """
    Ensure on-policy rollouts exist for any eval datasets listed in eval_on_policy_datasets.

    We generate locally (no API calls) and default to one rollout per prompt.
    """
    on_policy_datasets = config_dict.get("eval_on_policy_datasets") or []
    if not on_policy_datasets:
        return

    model_name = ModelName(config_dict["model_name"])
    repo = DatasetRepository()

    missing: list[str] = []
    for dataset_partial_id in on_policy_datasets:
        dataset_id = f"{dataset_partial_id}__{model_name.value}"
        if dataset_id not in repo.dataset_paths:
            missing.append(dataset_partial_id)

    if not missing:
        return

    logger.info(
        "Generating on-policy rollouts for datasets=%s with model=%s",
        ", ".join(missing),
        model_name.value,
    )
    model, tokenizer = dd_models.get_model_and_tokenizer(model_name, omit_model=False)
    def fallback_judgement(base_name: str, label_value: str) -> str:
        if base_name == "roleplaying":
            return "7" if label_value == "deceptive" else "1"
        return label_value

    for dataset_partial_id in missing:
        dataset = repo.get(dataset_partial_id, trim_reasoning=False)
        generator = RolloutGenerator.from_dialogue_dataset(dataset, model, tokenizer)
        generator.generate(num_rollouts=1)

        for rollout in generator.rollout_dialogues:
            grade = fallback_judgement(dataset.base_name, rollout.expected_label.value)
            rollout.graded_deception = [grade] * len(rollout.output_messages)
            rollout.grader_reasoning = ["Expected deception value"] * len(
                rollout.output_messages
            )

        out_file = (
            DD_REPO_ROOT
            / "data"
            / "rollouts"
            / f"{dataset_partial_id}__{model_name.value}.json"
        )
        out_file.parent.mkdir(parents=True, exist_ok=True)
        generator.save(out_file)
        repo.add(dataset_partial_id, model_name.value, str(out_file.relative_to(DD_REPO_ROOT)))
        generator.print_grading_statistics()


# -----------------------------------------------------------------------------
# CLI entrypoints
# -----------------------------------------------------------------------------


def run(
    config_file: str | Path = DEFAULT_CONFIG,
    **kwargs: dict[str, Any],
):
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config_dict = _load_config(config_path)
    config_dict |= kwargs
    _maybe_set_all_layers(config_dict)
    _ensure_on_policy_rollouts(config_dict)

    config = ExperimentConfig(**config_dict)  # type: ignore[reportGeneralTypeIssues]

    try:
        experiment = Experiment(config)
        if os.getenv("APOLLO_SKIP_DETECTION_MASKS", "1") != "1":
            save_detection_masks(experiment)
        experiment.get_detector()
    except Exception as err:
        # Clean up empty results folder on crash during detector construction
        for file_path in config.folder.iterdir():
            file_path.unlink()
        config.folder.rmdir()
        raise err

    experiment.get_eval_scores()
    experiment.get_control_scores()

    results(str(experiment.cfg.folder), scoring_method="max")


def rerun(folder: str, scoring_method: Literal["mean", "max", "relu"] = "mean"):
    cfg = ExperimentConfig.from_path(Path(folder))
    experiment = Experiment(cfg)
    experiment.get_detector()
    experiment.get_eval_scores()
    experiment.get_control_scores()
    results(str(experiment.cfg.folder), scoring_method=scoring_method)


def recompute_dataset(folder: str, dataset_name: str, shard_size: int = 12):
    cfg = ExperimentConfig.from_path(Path(folder))
    experiment = Experiment(cfg)
    experiment.recompute_scores_on_dataset(dataset_name, shard_size=shard_size)
    results(str(experiment.cfg.folder), scoring_method="mean")


def results(
    folder: str,
    scoring_method: PromptScorerMethod = "max",
    include_black_box: bool = False,
    generate_ai_liar_token_scores: bool = True,
):
    """
    Postprocess results and attach in-memory diagnostics.

    This:
      - patches dataset-specific plots to avoid crashing on missing columns;
      - runs the usual ExperimentPostprocessor;
      - optionally writes token-score HTML for ai_liar__original_without_answers;
      - builds a diagnostics CSV directly from the in-memory results DataFrame,
        marking rows where AUROC / counts are clearly problematic.
    """
    folder_path = Path(folder)
    cfg = ExperimentConfig.from_path(folder_path)
    experiment = Experiment(cfg)

    # Ensure eval scores are available for postprocessing (do not re-compute control).
    experiment.get_eval_scores(all_tokens=False)

    # Patch plotting to tolerate missing columns for ai_liar / roleplaying.
    _patch_postprocess_plotting()

    postprocessor = ExperimentPostprocessor(
        experiment,
        scoring_method=scoring_method,
        include_black_box=include_black_box,
    )

    try:
        postprocessor.run()
    except Exception as e:
        # We already patch known issues, but keep this as a final guard.
        logger.warning("Postprocess crashed (%s); continuing with diagnostics.", e)

    if generate_ai_liar_token_scores:
        _write_token_scores_for_eval_dataset(
            experiment,
            postprocessor.scorer,
            "ai_liar__original_with_answers",
            num_samples=20,
        )

    # In-memory diagnostics, no JSON parsing.
    _write_results_diagnostics_in_memory(folder_path, postprocessor)


def just_black_box_scores(folder: str):
    cfg = ExperimentConfig.from_path(Path(folder))
    experiment = Experiment(cfg)
    experiment.get_black_box_scores()


def _write_token_scores_for_eval_dataset(
    experiment: Experiment,
    scorer: PromptScorer,
    dataset_name: str,
    num_samples: int = 20,
) -> None:
    try:
        from deception_detection.results import token_html  # type: ignore
    except Exception as e:
        logger.warning("Token-score plotting unavailable (%s).", e)
        return

    eval_dataset_names = {dataset.name for dataset in experiment.eval_datasets}
    if dataset_name not in eval_dataset_names:
        logger.info("Skipping token scores for %s (not in eval_data).", dataset_name)
        return

    try:
        token_html.plot_eval_token_scores_for_dataset(
            experiment,
            scorer,
            dataset_name,
            num_samples=num_samples,
        )
    except Exception as e:
        logger.warning("Failed to write token scores for %s: %s", dataset_name, e)


if __name__ == "__main__":
    fire.Fire(
        {
            "run": run,
            "results": results,
            "rerun": rerun,
            "recompute": recompute_dataset,
            "black_box": just_black_box_scores,
        }
    )
