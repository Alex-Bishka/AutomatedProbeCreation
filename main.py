import os
import fire
import numpy as np
import pandas as pd
from logger import logger
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix

from llm import load_model
from paths import DATA_DIR
from probe import get_last_token_activations, train_probe

from parsing_agent import parse_user_input
from jailbreak_agent import test_jailbreak, generate_jailbreak_prompts


def load_cache(cache_path: str):
    assert os.path.exists(os.path.join(DATA_DIR, cache_path)), f"Cache path {cache_path} does not exist"
    logger.info(f"Loading cached dataset from {cache_path}")
    train_data = pd.read_csv(os.path.join(DATA_DIR, cache_path, "train.csv"))
    test_data = pd.read_csv(os.path.join(DATA_DIR, cache_path, "test.csv"))

    train_set_true = train_data[train_data["label"] == True]["prompt"].tolist()
    train_set_false = train_data[train_data["label"] == False]["prompt"].tolist()

    test_set_true = test_data[test_data["label"] == True]["prompt"].tolist()
    test_set_false = test_data[test_data["label"] == False]["prompt"].tolist()

    return train_set_true, train_set_false, test_set_true, test_set_false

# python3 main.py --cache_path="jailbreak_cache_v1"
# python3 main.py --cache_path="currated_cache_v1"
def main(
    judge_model_name: str = "google/gemini-2.5-flash-lite",
    jailbreak_model_name: str = "meta-llama/llama-3.2-1b-instruct",
    attack_model_name: str = "google/gemini-2.5-flash-lite",
    translator_model_name: str = "google/gemini-2.5-flash-lite",
    train_percentage: float = 0.9,
    cache_path: str | None = None,
    threshold: float = 0.5
):
    if cache_path and os.path.exists(os.path.join(DATA_DIR, cache_path)):
        train_set_true, train_set_false, test_set_true, test_set_false = load_cache(cache_path)
    else:
        user_input = input("Enter a concept you want to test the model on: ")
        behavior, reference = parse_user_input(user_input=user_input, translator_model_name=translator_model_name)
        logger.info("Parsed behavior:")
        logger.info(behavior)
        logger.info("Parsed reference:")
        logger.info(reference)

        attack_prompt = test_jailbreak(
            behavior=behavior,
            reference=reference,
            attack_llm_name=attack_model_name,
            jailbreak_model_name=jailbreak_model_name,
            judge_model_name=judge_model_name,
            max_attempts=5
        )

        logger.info("Final attack prompt:")
        logger.info(attack_prompt)
        user_confirmation = input("Are you okay with this attack prompt? (y/n): ").strip().lower()
        if user_confirmation != 'y':
            logger.info("User declined the attack prompt. Exiting.")
            print("Attack prompt declined. Exiting.")
            exit()

        logger.info("User confirmed the attack prompt. Continuing...")

        if attack_prompt == "":
            raise Exception("Failed to jailbreak the model")
        
        jailbreak_success, jailbreak_failure = generate_jailbreak_prompts(
            pair_model_name=attack_model_name,
            jailbreak_model_name=jailbreak_model_name,
            judge_model_name=judge_model_name,
            attack_prompt_clean=attack_prompt
        )
        
        train_set_true = jailbreak_success[:int(train_percentage*len(jailbreak_success))]
        test_set_true = jailbreak_success[int(train_percentage*len(jailbreak_success)):]

        train_set_false = jailbreak_failure[:int(train_percentage*len(jailbreak_failure))]
        test_set_false = jailbreak_failure[int(train_percentage*len(jailbreak_failure)):]

        train_data = pd.DataFrame({
            "prompt": train_set_true + train_set_false,
            "label": [True] * len(train_set_true) + [False] * len(train_set_false)
        })

        test_data = pd.DataFrame({
            "prompt": test_set_true + test_set_false,
            "label": [True] * len(test_set_true) + [False] * len(test_set_false)
        })

        if cache_path is not None:
            output_dir = f"{DATA_DIR}/{cache_path}"
            os.makedirs(output_dir, exist_ok=True)
            train_data.to_csv(os.path.join(output_dir, "train.csv"), index=False)
            test_data.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    model, tokenizer = load_model(jailbreak_model_name)
    probe = train_probe(
        dataset_true=train_set_true, 
        dataset_false=train_set_false, 
        model=model, 
        tokenizer=tokenizer
    )

    layer = probe.layer

    true_activations = get_last_token_activations(
        model=model,
        tokenizer=tokenizer,
        dataset = test_set_true
    )[layer]

    false_activations = get_last_token_activations(
        model=model,
        tokenizer=tokenizer,
        dataset = test_set_false
    )[layer]

    # Get raw scores (not probabilities)
    true_preds = probe.predict(true_activations).cpu().numpy()
    false_preds = probe.predict(false_activations).cpu().numpy()

    # Combine predictions and labels for ROC-AUC
    all_preds = np.concatenate([true_preds, false_preds])
    all_labels = np.concatenate([np.ones(len(true_preds)), np.zeros(len(false_preds))])

    # Calculate ROC-AUC (threshold-independent metric)
    auc = roc_auc_score(all_labels, all_preds)

    # Find optimal threshold using Youden's J statistic
    fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # Calculate metrics at both optimal and user-specified thresholds
    logger.info(f"Model tested: {jailbreak_model_name}")
    logger.info(f"Layer: {layer}\n")

    logger.info(f"ROC-AUC Score: {auc:.4f}\n")

    logger.info(f"=== Metrics at optimal threshold ({optimal_threshold:.4f}) ===")
    optimal_preds = (all_preds > optimal_threshold).astype(int)
    optimal_acc = accuracy_score(all_labels, optimal_preds)
    optimal_cm = confusion_matrix(all_labels, optimal_preds)
    logger.info(f"Overall accuracy: {optimal_acc:.4f}")
    logger.info(f"True Positives: {optimal_cm[1,1]}, False Positives: {optimal_cm[0,1]}")
    logger.info(f"True Negatives: {optimal_cm[0,0]}, False Negatives: {optimal_cm[1,0]}\n")

    logger.info(f"=== Metrics at user threshold ({threshold}) ===")
    user_preds = (all_preds > threshold).astype(int)
    user_acc = accuracy_score(all_labels, user_preds)
    user_cm = confusion_matrix(all_labels, user_preds)
    logger.info(f"Overall accuracy: {user_acc:.4f}")
    logger.info(f"Accuracy on true set: {((true_preds > threshold).sum() / len(true_preds)):.4f}")
    logger.info(f"Accuracy on false set: {((false_preds < threshold).sum() / len(false_preds)):.4f}")
    logger.info(f"True Positives: {user_cm[1,1]}, False Positives: {user_cm[0,1]}")
    logger.info(f"True Negatives: {user_cm[0,0]}, False Negatives: {user_cm[1,0]}\n")

    logger.info(f"Average score on true set: {true_preds.mean():.4f}")
    logger.info(f"Average score on false set: {false_preds.mean():.4f}")
    logger.info(f"Difference: {true_preds.mean() - false_preds.mean():.4f}\n")

    logger.info("True set scores: %s", [f"{x:.4f}" for x in true_preds.tolist()])
    logger.info("False set scores: %s", [f"{x:.4f}" for x in false_preds.tolist()])
    

if __name__ == "__main__":
    fire.Fire(main)