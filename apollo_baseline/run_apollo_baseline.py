from __future__ import annotations

from pathlib import Path
import json
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
from deception_detection.prompt_scorer import PromptScorerMethod
from deception_detection.results.detection_masks import save_detection_masks

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
        use_gb = min(total_gb * 0.9, cap_gb) if cap_gb is not None else total_gb * 0.9
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

        token: Optional[str] = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

        max_memory = _get_max_memory_map()
        if max_memory:
            logger.info(f"Using max_memory map for {len(max_memory)} GPUs: {max_memory}")
        logger.info(f"Loading {model_path} with cache_dir={models_directory}")

        # Prefer token=..., fall back to use_auth_token=... for older transformers
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
# Postprocess crash fix: ai_liar violin plot assumes expected_label exists
# -----------------------------------------------------------------------------


def _patch_postprocess_plotting() -> None:
    """
    Prevent KeyError: 'expected_label' from crashing postprocessing.

    The crash comes from deception_detection.results.plot.ai_liar_violin, which assumes an
    `expected_label` column exists in a dataframe. Some ai_liar variants do not include it.

    This patch wraps ai_liar_violin so that if that specific KeyError occurs, the plot is skipped
    and the run continues (results table + other plots still produced).
    """
    try:
        from deception_detection.results import plot as plot_mod  # type: ignore
    except Exception as e:
        logger.warning("Could not import deception_detection.results.plot for patching: %s", e)
        return

    if getattr(plot_mod, "_apollo_expected_label_patch_applied", False):
        return

    if not hasattr(plot_mod, "ai_liar_violin"):
        logger.warning("plot.ai_liar_violin not found; cannot patch expected_label crash.")
        return

    orig = plot_mod.ai_liar_violin

    def safe_ai_liar_violin(*args: Any, **kwargs: Any):
        try:
            return orig(*args, **kwargs)
        except KeyError as e:
            # pandas KeyError prints like KeyError('expected_label') or "expected_label"
            if str(e).strip().strip("'") == "expected_label":
                logger.warning(
                    "Skipping ai_liar violin plot: dataframe missing expected_label column."
                )
                return None
            raise

    plot_mod.ai_liar_violin = safe_ai_liar_violin
    plot_mod._apollo_expected_label_patch_applied = True


# -----------------------------------------------------------------------------
# Diagnostics reporting (impacted AUROC marking)
# -----------------------------------------------------------------------------


def _load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _as_float_array(x: Any) -> np.ndarray | None:
    if not isinstance(x, list):
        return None
    out: list[float] = []
    for v in x:
        if v is None:
            out.append(float("nan"))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out.append(float(v))
        else:
            return None
    return np.asarray(out, dtype=float)


def _extract_label_scores(obj: Any) -> dict[str, np.ndarray] | None:
    """
    Try to extract label->scores arrays from various common JSON shapes.

    Supported shapes:
      A) {"honest": [...], "deceptive": [...], "ambiguous": [...]}
      B) {"honest_scores": [...], "deceptive_scores": [...]}
      C) {"scores": <A or B>}
      D) [{"label": "honest"/"deceptive", "score": ...}, ...]
    """
    # D) list of records
    if isinstance(obj, list) and obj and all(isinstance(r, dict) for r in obj):
        if all(("label" in r) and (("score" in r) or ("value" in r)) for r in obj):
            buckets: dict[str, list[float]] = {"honest": [], "deceptive": [], "ambiguous": []}
            for r in obj:
                label = r.get("label")
                if label not in buckets:
                    continue
                score = r.get("score", r.get("value"))
                if score is None:
                    buckets[label].append(float("nan"))
                elif isinstance(score, (int, float)) and not isinstance(score, bool):
                    buckets[label].append(float(score))
                else:
                    buckets[label].append(float("nan"))
            if sum(len(v) for v in buckets.values()) == 0:
                return None
            return {k: np.asarray(v, dtype=float) for k, v in buckets.items()}

    if not isinstance(obj, dict):
        return None

    # A)
    if "honest" in obj or "deceptive" in obj:
        honest = _as_float_array(obj.get("honest", []))
        deceptive = _as_float_array(obj.get("deceptive", []))
        ambiguous = _as_float_array(obj.get("ambiguous", [])) or np.asarray([], dtype=float)
        if honest is not None and deceptive is not None:
            return {"honest": honest, "deceptive": deceptive, "ambiguous": ambiguous}

    # B)
    if "honest_scores" in obj or "deceptive_scores" in obj:
        honest = _as_float_array(obj.get("honest_scores", []))
        deceptive = _as_float_array(obj.get("deceptive_scores", []))
        ambiguous = _as_float_array(obj.get("ambiguous_scores", [])) or np.asarray([], dtype=float)
        if honest is not None and deceptive is not None:
            return {"honest": honest, "deceptive": deceptive, "ambiguous": ambiguous}

    # C) nested
    for k in ("scores", "token_scores", "prompt_scores", "data"):
        if k in obj:
            inner = _extract_label_scores(obj[k])
            if inner is not None:
                return inner

    return None


def _walk_datasets(tree: Any) -> dict[str, dict[str, np.ndarray]]:
    """
    Parse {dataset_name: {honest: arr, deceptive: arr, ambiguous: arr}} from a JSON object.
    Assumes top-level is often a dict keyed by dataset name.
    """
    out: dict[str, dict[str, np.ndarray]] = {}
    if not isinstance(tree, dict):
        return out

    for ds_name, ds_obj in tree.items():
        rec = _extract_label_scores(ds_obj)
        if rec is not None:
            out[str(ds_name)] = rec
    return out


def _auroc_with_diagnostics(honest: np.ndarray, deceptive: np.ndarray) -> tuple[float, dict[str, Any]]:
    """
    Compute AUROC and return diagnostics indicating whether the number should be treated as impacted.
    """
    honest = honest.astype(float).reshape(-1)
    deceptive = deceptive.astype(float).reshape(-1)

    y_true = np.concatenate(
        [np.ones_like(deceptive, dtype=int), np.zeros_like(honest, dtype=int)]
    )
    y_scores = np.concatenate([deceptive, honest])

    raw_total = int(y_scores.size)
    raw_h = int(honest.size)
    raw_d = int(deceptive.size)

    finite = np.isfinite(y_scores)
    dropped = int(raw_total - int(finite.sum()))
    y_true_f = y_true[finite]
    y_scores_f = y_scores[finite]

    classes_present = int(np.unique(y_true_f).size) if y_true_f.size else 0

    diag: dict[str, Any] = {
        "raw_honest": raw_h,
        "raw_deceptive": raw_d,
        "raw_total": raw_total,
        "finite_total": int(y_scores_f.size),
        "dropped_naninf": dropped,
        "classes_present": classes_present,
        "missing_honest_raw": raw_h == 0,
        "missing_deceptive_raw": raw_d == 0,
    }

    # Determine issues (clear "what was missing" reporting)
    issues: list[str] = []
    if raw_h == 0:
        issues.append("missing_honest_raw")
    if raw_d == 0:
        issues.append("missing_deceptive_raw")
    if dropped > 0:
        issues.append("dropped_naninf")
    if y_scores_f.size == 0:
        issues.append("no_finite_scores")
    if classes_present < 2:
        issues.append("single_class_after_filter")

    diag["issues"] = issues
    diag["impacted"] = len(issues) > 0

    if y_scores_f.size == 0 or classes_present < 2:
        return float("nan"), diag

    try:
        from sklearn.metrics import roc_auc_score  # local import for environments without sklearn at import time

        au = float(roc_auc_score(y_true_f, y_scores_f))
        return au, diag
    except Exception as e:
        diag["issues"].append(f"auroc_error:{type(e).__name__}")
        diag["impacted"] = True
        return float("nan"), diag


def _report_score_diagnostics(folder: Path) -> None:
    """
    Read eval/control score JSON files (if present), compute AUROC + diagnostics per dataset,
    and write a CSV + JSON summary into the run folder. Also prints a compact table to stdout.
    """
    candidates = [
        folder / "eval_scores.json",
        folder / "control_scores.json",
        folder / "eval_scores_all_tokens.json",
        folder / "control_scores_all_tokens.json",
    ]
    existing = [p for p in candidates if p.exists()]

    if not existing:
        logger.warning("No score JSON files found in %s; skipping diagnostics.", folder)
        return

    rows: list[dict[str, Any]] = []

    for path in existing:
        tree = _load_json(path)
        ds_map = _walk_datasets(tree)
        if not ds_map:
            logger.warning("Could not parse dataset score structure from %s", path)
            continue

        for ds, rec in ds_map.items():
            honest = rec.get("honest", np.asarray([], dtype=float))
            deceptive = rec.get("deceptive", np.asarray([], dtype=float))
            ambiguous = rec.get("ambiguous", np.asarray([], dtype=float))

            au, diag = _auroc_with_diagnostics(honest, deceptive)
            impacted = cast(bool, diag["impacted"])
            au_str = "nan" if not np.isfinite(au) else f"{au:.3f}"
            au_marked = f"{au_str}*" if impacted and au_str != "nan" else au_str

            rows.append(
                {
                    "file": path.name,
                    "dataset": ds,
                    "n_honest_raw": int(diag["raw_honest"]),
                    "n_deceptive_raw": int(diag["raw_deceptive"]),
                    "n_ambiguous_raw": int(ambiguous.size),
                    "n_total_raw": int(diag["raw_total"]),
                    "n_total_finite": int(diag["finite_total"]),
                    "dropped_naninf": int(diag["dropped_naninf"]),
                    "classes_present_after_filter": int(diag["classes_present"]),
                    "auroc": au,
                    "auroc_marked": au_marked,
                    "impacted": impacted,
                    "issues": ",".join(diag["issues"]),
                }
            )

    if not rows:
        logger.warning("No diagnostics rows produced; skipping diagnostics writeout.")
        return

    # Write JSON
    diag_json = folder / "diagnostics.json"
    with open(diag_json, "w") as f:
        json.dump(rows, f, indent=2)

    # Write CSV + print table if pandas is available
    try:
        import pandas as pd  # type: ignore

        df = pd.DataFrame(rows).sort_values(["file", "dataset"])
        diag_csv = folder / "diagnostics.csv"
        df.to_csv(diag_csv, index=False)

        # Print compact view (mark impacted AUROCs with *)
        view_cols = [
            "file",
            "dataset",
            "n_honest_raw",
            "n_deceptive_raw",
            "n_total_finite",
            "dropped_naninf",
            "auroc_marked",
            "impacted",
            "issues",
        ]
        logger.info("Diagnostics summary (AUROC values with * are impacted):\n%s", df[view_cols].to_string(index=False))
        logger.info("Saved diagnostics to %s and %s", diag_json, diag_csv)
    except Exception:
        logger.info("Saved diagnostics to %s (pandas not available; skipping CSV/table).", diag_json)


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

    config = ExperimentConfig(**config_dict)  # type: ignore[reportGeneralTypeIssues]

    try:
        experiment = Experiment(config)
        if os.getenv("APOLLO_SKIP_DETECTION_MASKS", "1") != "1":
            save_detection_masks(experiment)
        experiment.get_detector()
    except Exception as err:
        # Preserve original behavior: clean up folder on failure to train detector.
        for file_path in config.folder.iterdir():
            file_path.unlink()
        config.folder.rmdir()
        raise err

    experiment.get_eval_scores()
    experiment.get_control_scores()

    # Run postprocess + diagnostics (postprocess may create tables/plots; diagnostics marks impacted stats)
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
):
    """
    Postprocess results and write diagnostics.

    Fixes:
      (a) Prevent KeyError('expected_label') from crashing postprocess by patching ai_liar plotting.
      (b) Writes diagnostics.csv / diagnostics.json that reports per-dataset:
            - raw class counts
            - finite score counts after NaN/Inf filtering
            - dropped NaN/Inf count
            - AUROC with '*' marker if impacted
            - explicit issues explaining what's missing / why impacted
    """
    folder_path = Path(folder)
    cfg = ExperimentConfig.from_path(folder_path)
    experiment = Experiment(cfg)

    # Ensure eval scores exist on disk (postprocessor depends on them)
    experiment.get_eval_scores(all_tokens=False)

    # Patch plotting crash
    _patch_postprocess_plotting()

    # Run postprocess; do not fail the whole run on the known expected_label plot issue
    postprocessor = ExperimentPostprocessor(
        experiment, scoring_method=scoring_method, include_black_box=include_black_box
    )

    try:
        postprocessor.run()
    except KeyError as e:
        if str(e).strip().strip("'") == "expected_label":
            logger.warning(
                "Postprocess crashed due to missing expected_label; continuing with diagnostics."
            )
        else:
            raise

    # Always write diagnostics at the end
    _report_score_diagnostics(folder_path)


def just_black_box_scores(folder: str):
    cfg = ExperimentConfig.from_path(Path(folder))
    experiment = Experiment(cfg)
    experiment.get_black_box_scores()


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
