from pathlib import Path
import sys
from typing import Any, Literal

import fire
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DECEPTION_DET_ROOT = REPO_ROOT / "deception-detection"
if str(DECEPTION_DET_ROOT) not in sys.path:
    sys.path.insert(0, str(DECEPTION_DET_ROOT))

from deception_detection.experiment import Experiment, ExperimentConfig
from deception_detection.log import logger
from deception_detection.models import ModelName, get_num_layers_for_model
from deception_detection.postprocess import ExperimentPostprocessor
from deception_detection.prompt_scorer import PromptScorerMethod
from deception_detection.results.detection_masks import save_detection_masks

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "apollo_llama3_1_8b_all_layers.yaml"


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
