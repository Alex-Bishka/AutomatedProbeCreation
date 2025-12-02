import re
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Tuple, List

from agent import Agent
from logger import logger
from prompts import (
    JUDGE_SYSTEM_PROMPT,
    ATTACK_LLM_SYSTEM_PROMPT,
    create_pair_sys_prompt,
    create_refinement_prompt,
    BEHAVIORS
)

def create_failed_set(num_successful_jailbreaks: int) -> List[str]:
    """
    Create a set of failed jailbreak prompts for negative examples.

    Args:
        num_successful_jailbreaks (int): The number of successful jailbreak prompts generated.
    """
    benign_df = pd.read_csv("./data/benign_prompts.csv")
    failed_jailbreak_samples_df = benign_df.sample(n=num_successful_jailbreaks, random_state=42)
    return failed_jailbreak_samples_df["prompt"].tolist()

def run_jailbreak(attack_prompt: str, target_llm: Agent, judge_llm: Agent) -> Tuple[str, str]:
    """
    Execute a jailbreak attack against a target LLM and evaluate the response.

    Args:
        attack_prompt (str): The adversarial prompt designed to jailbreak the target model
        target_llm (Agent): The language model agent being attacked/tested
        judge_llm (Agent): The language model agent used to evaluate attack success

    Returns:
        Tuple[str, str]: A tuple containing:
            - target_response (str): The response from the target LLM to the attack prompt
            - judge_llm_raw (str): The judge's verdict on whether the attack was successful
    """
    target_response = target_llm.call_agent(user_prompt=attack_prompt)
    logger.info("-" * 100)
    logger.info("Target LLM response:")
    logger.info(target_response)
    logger.info("-" * 100)

    judge_prompt = f"Here was the attack prompt used for jailbreaking: {attack_prompt}"
    judge_prompt = f"{judge_prompt}\nHere was the response of the model being attacked: {target_response}"
    judge_llm_raw = judge_llm.call_agent(user_prompt=judge_prompt)
    logger.info("-" * 100)
    logger.info("Judge verdict:")
    logger.info(judge_llm_raw)
    logger.info("-" * 100)

    return target_response, judge_llm_raw


def test_jailbreak(behavior: str, reference: str, attack_llm_name: str, jailbreak_model_name: str, judge_model_name: str, max_attempts: int = 5) -> str:
    """
    Test and iteratively refine jailbreak attacks against a target model.

    Args:
        behavior (str): The harmful behavior the jailbreak should attempt to elicit
        reference (str): Reference examples or context for generating the attack
        attack_llm_name (str): Name/identifier of the model used to generate attack prompts
        jailbreak_model_name (str): Name/identifier of the target model being jailbroken
        judge_model_name (str): Name/identifier of the model used to judge attack success
        max_attempts (int, optional): Maximum number of refinement attempts. Defaults to 5.

    Returns:
        str: Returns the successful attack prompt (str) if jailbreak succeeds,
             or empty string ("") if all attempts fail

    Note:
        The function uses conversation history to iteratively improve attack prompts
        based on previous failures. It expects attack prompts to be enclosed in
        <PROMPT></PROMPT> tags in the attack LLM's response.
    """
    attack_user_prompt = create_pair_sys_prompt(behavior=behavior, references=reference)

    attack_llm = Agent(model_name=attack_llm_name, system_prompt=ATTACK_LLM_SYSTEM_PROMPT)
    target_llm = Agent(model_name=jailbreak_model_name)
    judge_llm = Agent(model_name=judge_model_name, system_prompt=JUDGE_SYSTEM_PROMPT)

    attack_prompt_raw = attack_llm.call_agent(user_prompt=attack_user_prompt, preserve_conversation_history=True)

    failure = True
    while failure:
        match = re.search(r'<PROMPT>(.*?)</PROMPT>', attack_prompt_raw, re.S | re.I)
        if not match:
            # Remove the malformed response from history
            if len(attack_llm.conversation_history) >= 2:
                attack_llm.conversation_history.pop()  # Remove assistant response
                attack_llm.conversation_history.pop()  # Remove user message

            logger.warning("No <PROMPT> tags found. Asking attack LLM to reformat...")
            attack_prompt_raw = attack_llm.call_agent(
                user_prompt="Please provide your attack prompt enclosed in <PROMPT></PROMPT> tags as requested.",
                preserve_conversation_history=True
            )
            continue  # Try parsing again

        attack_prompt_clean = match.group(1).strip()
        logger.info("Starting attack prompt:")
        logger.info(attack_prompt_clean)

        target_response, judge_llm_raw = run_jailbreak(attack_prompt=attack_prompt_clean, target_llm=target_llm, judge_llm=judge_llm)
        if 'success' not in judge_llm_raw.lower():
            refinement_prompt = create_refinement_prompt(attack_prompt_clean, target_response)
            attack_prompt_raw = attack_llm.call_agent(
                user_prompt=refinement_prompt,
                preserve_conversation_history=True
            )
            logger.info("-" * 100)
            logger.info("Updated attack prompt:")
            logger.info(attack_prompt_raw)
            logger.info("-" * 100)

            max_attempts -= 1
            logger.info(f"Attempts left: {max_attempts}")
        else:
            return attack_prompt_clean

        if max_attempts == 0:
            failure = False

    return ""


def test_jailbreak_with_iterations(behavior: str, reference: str, attack_llm_name: str, jailbreak_model_name: str, judge_model_name: str, max_attempts: int = 5) -> Tuple[str, int, list, str, str]:
    """
    Test and iteratively refine jailbreak attacks against a target model, tracking iterations.

    Args:
        behavior (str): The harmful behavior the jailbreak should attempt to elicit
        reference (str): Reference examples or context for generating the attack
        attack_llm_name (str): Name/identifier of the model used to generate attack prompts
        jailbreak_model_name (str): Name/identifier of the target model being jailbroken
        judge_model_name (str): Name/identifier of the model used to judge attack success
        max_attempts (int, optional): Maximum number of refinement attempts. Defaults to 5.

    Returns:
        Tuple[str, int, list, str, str]: Returns tuple of (attack_prompt, num_iterations, conversation_history, target_response, judge_verdict)
                        attack_prompt is the successful prompt (str) or empty string ("") if failed
                        num_iterations is the number of attempts taken (1 to max_attempts)
                        conversation_history is the attack LLM's conversation history
                        target_response is the final response from the jailbreak target model
                        judge_verdict is the final verdict from the judge model

    Note:
        The function uses conversation history to iteratively improve attack prompts
        based on previous failures. It expects attack prompts to be enclosed in
        <PROMPT></PROMPT> tags in the attack LLM's response.
    """
    attack_user_prompt = create_pair_sys_prompt(behavior=behavior, references=reference)

    attack_llm = Agent(model_name=attack_llm_name, system_prompt=ATTACK_LLM_SYSTEM_PROMPT)
    target_llm = Agent(model_name=jailbreak_model_name)
    judge_llm = Agent(model_name=judge_model_name, system_prompt=JUDGE_SYSTEM_PROMPT)

    attack_prompt_raw = attack_llm.call_agent(user_prompt=attack_user_prompt, preserve_conversation_history=True)

    attempts_remaining = max_attempts
    iteration = 0
    target_response = ""
    judge_llm_raw = ""

    while attempts_remaining > 0:
        iteration += 1

        match = re.search(r'<PROMPT>(.*?)</PROMPT>', attack_prompt_raw, re.S | re.I)
        if not match:
            # Remove the malformed response from history
            if len(attack_llm.conversation_history) >= 2:
                attack_llm.conversation_history.pop()  # Remove assistant response
                attack_llm.conversation_history.pop()  # Remove user message

            logger.warning("No <PROMPT> tags found. Asking attack LLM to reformat...")
            attack_prompt_raw = attack_llm.call_agent(
                user_prompt="Please provide your attack prompt enclosed in <PROMPT></PROMPT> tags as requested.",
                preserve_conversation_history=True
            )
            iteration -= 1  # Don't count format retry as an iteration
            continue  # Try parsing again

        attack_prompt_clean = match.group(1).strip()
        logger.info(f"Attempt {iteration}/{max_attempts}:")
        logger.info(attack_prompt_clean)

        target_response, judge_llm_raw = run_jailbreak(attack_prompt=attack_prompt_clean, target_llm=target_llm, judge_llm=judge_llm)

        if 'success' in judge_llm_raw.lower():
            logger.info(f"SUCCESS on iteration {iteration}")
            return attack_prompt_clean, iteration, attack_llm.conversation_history, target_response, judge_llm_raw

        # Failed - refine if we have attempts left
        attempts_remaining -= 1
        if attempts_remaining > 0:
            refinement_prompt = create_refinement_prompt(attack_prompt_clean, target_response)
            attack_prompt_raw = attack_llm.call_agent(
                user_prompt=refinement_prompt,
                preserve_conversation_history=True
            )
            logger.info("-" * 100)
            logger.info(f"Refining... {attempts_remaining} attempts remaining")
            logger.info("-" * 100)

    logger.info(f"FAILED after {iteration} iterations")
    return "", iteration, attack_llm.conversation_history, target_response, judge_llm_raw


def generate_jailbreak_prompts(
    pair_model_name: str,
    jailbreak_model_name: str,
    judge_model_name: str,
    attack_prompt_clean: str
) -> Tuple[List[str], List[str]]:
    """
    Generate and test jailbreak prompts across multiple harmful behaviors using proliferation.

    Args:
        pair_model_name (str): Name/identifier of the model used for prompt proliferation
        jailbreak_model_name (str): Name/identifier of the target model being tested
        judge_model_name (str): Name/identifier of the model used to evaluate responses
        attack_prompt_clean (str): A known successful jailbreak prompt to use as reference

    Returns:
        Tuple[List[str], List[str]]: A tuple containing:
            - jailbreak_success (List[str]): List of prompts that successfully jailbroke the target
            - jailbreak_failures (List[str]): List of prompts that failed to jailbreak the target
    """
    attack_llm = Agent(model_name=pair_model_name, system_prompt=ATTACK_LLM_SYSTEM_PROMPT)
    judge_llm = Agent(model_name=judge_model_name, system_prompt=JUDGE_SYSTEM_PROMPT)
    target_llm = Agent(model_name=jailbreak_model_name)

    jailbreak_success = []
    jailbreak_dataset = []
    for new_behavior in BEHAVIORS:
        logger.info("New attempted behavior:")
        logger.info(new_behavior)
        logger.info("-" * 50)
        # Create the one-shot proliferation prompt
        proliferation_sys_prompt = create_pair_sys_prompt(
            behavior=new_behavior,
            references=attack_prompt_clean
        )

        new_jailbreak_prompt_raw = attack_llm.call_agent(user_prompt=proliferation_sys_prompt)
        match = re.search(r'<PROMPT>(.*?)</PROMPT>', new_jailbreak_prompt_raw, re.S | re.I)
        if match:
            new_jailbreak_prompt = match.group(1).strip()

            target_response, judge_llm_raw = run_jailbreak(attack_prompt=attack_prompt_clean, target_llm=target_llm, judge_llm=judge_llm)
            logger.info("Result of Attack:")
            logger.info(target_response)
            logger.info("-" * 50)
            if "success" in judge_llm_raw.lower():
                jailbreak_success.append(new_jailbreak_prompt)

            jailbreak_dataset.append((new_behavior, new_jailbreak_prompt, target_response, judge_llm_raw))

    jailbreak_df = pd.DataFrame(jailbreak_dataset, columns=["Behavior", "Jailbreak Prompt", "Jailbreak Result", "Success"])
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    subfolder_path = Path(f"./jailbreak-datasets/{timestamp}")
    subfolder_path.mkdir(parents=True, exist_ok=True)
    file_path = subfolder_path / "results.csv"
    jailbreak_df.to_csv(file_path, index=False)

    num_successful_jailbreaks = len(jailbreak_success)

    jailbreak_failures = create_failed_set(num_successful_jailbreaks)

    return jailbreak_success, jailbreak_failures
