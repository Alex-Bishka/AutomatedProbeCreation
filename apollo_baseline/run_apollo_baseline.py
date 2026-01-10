from pathlib import Path
import os
import sys
from typing import Any, Literal, Optional

import fire
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
    def _patched_get_llama3_model_and_tokenizer(
        model_name: ModelName, omit_model: bool = False
    ):
        # Keep existing tokenizer logic (if dd_models internally uses HF, it should respect env vars)
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

        # IMPORTANT FIX:
        # Never hardcode /data. Use HF_HOME / scratch / home cache instead.
        cache_root = _resolve_hf_cache_dir()
        models_directory = cache_root / "transformers"
        _ensure_dir(models_directory)

        # If HF token is present (for gated models), pass it explicitly.
        # transformers supports 'token='; some older versions use 'use_auth_token='.
        token: Optional[str] = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

        max_memory = _get_max_memory_map()
        if max_memory:
            logger.info(f"Using max_memory map for {len(max_memory)} GPUs: {max_memory}")
        logger.info(f"Loading {model_path} with cache_dir={models_directory}")

        # Prefer token=..., but fall back to use_auth_token if needed.
        # (We don't know exactly which transformers version is installed across systems.)
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


def _load_config(config_file: Path) -> dict[str, Any]:
    with open(config_file) as file:
        return yaml.safe_load(file)


def _maybe_set_all_layers(config_dict: dict[str, Any]) -> None:
    detect_layers = config_dict.get("detect_layers")
    if detect_layers:
        return

    model_name = ModelName(config_dict["model_name"])
    n_layers = get_num_layers_for_model(model_name)
    config_dict["detect_layers"] = list(range(n_layers))


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
    folder: str, scoring_method: PromptScorerMethod = "max", include_black_box: bool = False
):
    cfg = ExperimentConfig.from_path(Path(folder))
    experiment = Experiment(cfg)
    experiment.get_eval_scores(all_tokens=False)
    postprocessor = ExperimentPostprocessor(
        experiment, scoring_method=scoring_method, include_black_box=include_black_box
    )
    postprocessor.run()


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
