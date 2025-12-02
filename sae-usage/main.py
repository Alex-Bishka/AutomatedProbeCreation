#!/usr/bin/env python3
"""SAE-guided automated probe creation pipeline with interactive human checkpoints."""

import argparse
import json
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from src import (
    load_config,
    load_env_vars,
    create_output_dir,
    LLMAgent,
    ModelManager,
    GoodfireClient,
    SAEManager,
    save_yaml
)
from src.neuronpedia_client import NeuronpediaClient
from src.on_policy_generator import OnPolicyGenerator, SteeringConfig
from src.probe_trainer import ProbeTrainer, compute_steering_alignment
from src.degeneracy_detector import is_degenerate, compare_outputs
from src.interactive_checkpoints import (
    prompt_human_approval,
    display_sae_feature_results,
    display_amplification_tests,
    display_contrastive_pairs,
    display_probe_results,
    display_progress
)
from src.quality_validator import QualityValidator
from src.data_generator import ContrastivePair


# ==============================================================================
# PHASE 1: SAE FEATURE SEARCH & APPROVAL
# ==============================================================================

def phase1_sae_feature_search_approval(
    config: Dict,
    llm_agent: LLMAgent,
    neuronpedia_client: NeuronpediaClient,
    sae_manager: SAEManager
) -> tuple[int, str, np.ndarray, str]:
    """
    Search for SAE features and get LLM approval.

    Returns:
        Tuple of (feature_index, feature_label, decoder_direction, final_concept)
    """
    print("\n" + "="*60)
    print("PHASE 1: SAE FEATURE SEARCH & APPROVAL")
    print("="*60)

    concept = config['concept']['positive_class']
    max_attempts = config['steering']['max_feature_attempts']
    max_refinements = config['steering']['max_concept_refinements']

    refinement_count = 0
    current_concept = concept

    while refinement_count <= max_refinements:
        print(f"\nSearching for features matching: '{current_concept}'")

        # Search Neuronpedia
        search_results = neuronpedia_client.search_features(
            query=current_concept,
            top_k=max_attempts
        )

        if not search_results:
            print(f"No features found for '{current_concept}'")
            if refinement_count < max_refinements:
                print("Will try refining concept...")
                refinement_count += 1
                continue
            else:
                raise ValueError(f"No SAE features found after {max_refinements} refinements")

        # Get feature details (labels + examples)
        features_with_details = []
        for result in search_results[:max_attempts]:
            feature_idx = result['index']
            try:
                feature_data = neuronpedia_client.get_feature(feature_idx)

                # Extract examples from full_data
                examples = feature_data.get('full_data', {}).get('activations', [])

                # Build feature info
                feature_info = {
                    'index': feature_idx,
                    'label': result.get('description', feature_data.get('description', f'Feature {feature_idx}')),
                    'score': result.get('cosine_similarity', 0.0),
                    'examples': examples[:3]  # Top 3 examples
                }

                features_with_details.append(feature_info)
                print(f"  Feature {feature_idx}: score={feature_info['score']:.4f}, examples={len(examples)}")

            except Exception as e:
                print(f"Warning: Failed to get details for feature {feature_idx}: {e}")
                continue

        if not features_with_details:
            print("Could not retrieve feature details")
            if refinement_count < max_refinements:
                refinement_count += 1
                continue
            else:
                raise ValueError("Failed to retrieve feature details")

        # Display features
        display_sae_feature_results(features_with_details)

        # Try each feature with LLM approval
        approved_feature = None
        for i, feature in enumerate(features_with_details, 1):
            print(f"\n--- Testing Feature {i}/{len(features_with_details)} ---")
            print(f"Index: {feature['index']}")
            print(f"Label: {feature['label']}")
            print(f"Score: {feature['score']:.4f}")
            print(f"Examples: {len(feature['examples'])} activations")

            # Get LLM approval
            examples_text = [f"{ex}" for ex in feature['examples']]
            approved, reasoning = llm_agent.approve_sae_feature(
                concept=current_concept,
                label=feature['label'],
                examples=examples_text
            )

            print(f"LLM Decision: {'✓ APPROVED' if approved else '✗ REJECTED'}")
            print(f"Reasoning: {reasoning}")

            if approved:
                approved_feature = feature
                break

        # If feature approved, return it
        if approved_feature:
            feature_idx = approved_feature['index']
            decoder_direction = sae_manager.get_decoder_direction(feature_idx)

            if decoder_direction is None:
                raise ValueError(f"Failed to get decoder direction for feature {feature_idx}")

            print(f"\n✓ Selected Feature {feature_idx}: {approved_feature['label']}")
            return feature_idx, approved_feature['label'], decoder_direction, current_concept

        # All features rejected - refine concept
        if refinement_count < max_refinements:
            print(f"\nAll {len(features_with_details)} features rejected")
            print("Asking LLM to suggest nearest concept...")

            rejected_labels = [f['label'] for f in features_with_details]
            refined_concept = llm_agent.suggest_nearest_concept(
                original_concept=current_concept,
                rejected_features=rejected_labels
            )

            print(f"LLM suggests refined concept: '{refined_concept}'")

            # Human checkpoint
            approved = prompt_human_approval(
                message=f"Original concept '{current_concept}' didn't match SAE features.\nLLM suggests trying '{refined_concept}' instead.",
                data={'original': current_concept, 'refined': refined_concept},
                require_confirmation=True
            )

            if not approved:
                raise ValueError("User rejected concept refinement")

            current_concept = refined_concept
            refinement_count += 1
        else:
            raise ValueError(f"Failed to find suitable SAE feature after {max_refinements} refinements")

    raise ValueError("Exhausted all refinement attempts")


# ==============================================================================
# PHASE 2: AMPLIFICATION TUNING
# ==============================================================================

def phase2_amplification_tuning(
    config: Dict,
    llm_agent: LLMAgent,
    model_manager: ModelManager,
    steering_vector: np.ndarray,
    layer: int,
    concept: str
) -> float:
    """
    Tune steering amplification strength.

    Returns:
        Optimal amplification value
    """
    print("\n" + "="*60)
    print("PHASE 2: AMPLIFICATION TUNING")
    print("="*60)

    amplification = config['steering']['initial_amplification']
    min_amp, max_amp = config['steering']['amplification_range']
    adjustment_step = config['steering']['adjustment_step']
    num_test_prompts = config['steering']['num_test_prompts']
    check_degeneracy = config['steering']['check_degeneracy']

    # Generate test prompts
    print(f"\nGenerating {num_test_prompts} test prompts...")
    test_prompts = []
    for i in range(num_test_prompts):
        try:
            question = llm_agent.generate_question(
                positive_class=config['concept']['positive_class'],
                negative_class=config['concept']['negative_class'],
                topic=None
            )
            test_prompts.append(question)
        except Exception as e:
            print(f"Warning: Failed to generate test prompt {i+1}: {e}")

    if not test_prompts:
        raise ValueError("Failed to generate any test prompts")

    print(f"Generated {len(test_prompts)} test prompts")

    # Tuning loop
    max_iterations = 5
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1}: Testing amplification = {amplification} ---")

        # Generate test outputs
        test_results = []
        for prompt in test_prompts:
            conversation = [{"role": "user", "content": prompt}]

            # Steered generation
            try:
                steered = model_manager.generate_with_steering(
                    messages=conversation,
                    feature_direction=steering_vector,
                    layer=layer,
                    steering_strength=amplification,
                    max_new_tokens=200,
                    temperature=0.7
                )
            except Exception as e:
                print(f"Warning: Steered generation failed: {e}")
                steered = "[FAILED]"

            # Unsteered generation
            try:
                unsteered = model_manager.generate_response(
                    messages=conversation,
                    max_new_tokens=200,
                    temperature=0.7
                )
            except Exception as e:
                print(f"Warning: Unsteered generation failed: {e}")
                unsteered = "[FAILED]"

            # Check for degeneracy
            degenerate = False
            if check_degeneracy and steered != "[FAILED]":
                comparison = compare_outputs(steered, unsteered)
                degenerate = comparison['steered_degenerate']

            test_results.append({
                'prompt': prompt,
                'steered': steered,
                'unsteered': unsteered,
                'degenerate': degenerate
            })

        # Display results
        display_amplification_tests(test_results, amplification)

        # LLM judge
        verdict, reasoning, suggested_amp = llm_agent.judge_amplification(
            concept=concept,
            amplification=amplification,
            test_results=test_results
        )

        print(f"\nLLM Verdict: {verdict}")
        print(f"Reasoning: {reasoning}")

        if verdict == "APPROPRIATE":
            print(f"\n✓ Amplification tuned: {amplification}")
            break
        elif verdict == "TOO_STRONG":
            new_amp = max(amplification - adjustment_step, min_amp)
            if new_amp == amplification:
                print(f"⚠️  Cannot reduce amplification further (already at minimum: {min_amp})")
                print("Consider decreasing amplification_range minimum in config if steering is still too strong")
                break
            print(f"Reducing amplification: {amplification} -> {new_amp}")
            amplification = new_amp
        elif verdict == "TOO_WEAK":
            new_amp = min(amplification + adjustment_step, max_amp)
            if new_amp == amplification:
                print(f"⚠️  Cannot increase amplification further (already at maximum: {max_amp})")
                print("Consider increasing amplification_range maximum in config if steering is still too weak")
                break
            print(f"Increasing amplification: {amplification} -> {new_amp}")
            amplification = new_amp
        else:
            print(f"Unknown verdict: {verdict}")
            break

        if iteration == max_iterations - 1:
            print(f"\nReached max iterations, using amplification = {amplification}")

    # Human checkpoint
    approved = prompt_human_approval(
        message=f"Amplification tuning complete.\nSelected amplification: {amplification}\nPlease review the test outputs above.",
        data={'amplification': amplification, 'verdict': verdict},
        require_confirmation=True
    )

    if not approved:
        raise ValueError("User rejected amplification tuning results")

    return amplification


# ==============================================================================
# PHASE 3: QUESTION/STATEMENT GENERATION
# ==============================================================================

def phase3_question_generation(
    config: Dict,
    llm_agent: LLMAgent,
    num_pairs_target: int
) -> List[str]:
    """
    Generate questions/statements for contrastive pair generation.

    Returns:
        List of prompts
    """
    print("\n" + "="*60)
    print("PHASE 3: QUESTION/STATEMENT GENERATION")
    print("="*60)

    format_type = config['generation']['formats'][0] if config['generation']['formats'] else "qa"
    positive_class = config['concept']['positive_class']
    negative_class = config['concept']['negative_class']

    # Generate first batch
    print(f"\nGenerating initial batch of 10 {format_type} prompts...")
    prompts = []
    for i in range(10):
        try:
            if format_type == "qa":
                prompt = llm_agent.generate_question(
                    positive_class=positive_class,
                    negative_class=negative_class,
                    topic=None
                )
            else:  # statement
                prompt = llm_agent.generate_statement(
                    positive_class=positive_class,
                    negative_class=negative_class,
                    is_positive=True,
                    topic=None
                )
            prompts.append(prompt)
        except Exception as e:
            print(f"Warning: Failed to generate prompt {i+1}: {e}")

    if not prompts:
        raise ValueError("Failed to generate any prompts")

    # Display first batch
    print(f"\nGenerated {len(prompts)} prompts:")
    for i, prompt in enumerate(prompts[:5], 1):
        print(f"  {i}. {prompt}")
    if len(prompts) > 5:
        print(f"  ... and {len(prompts) - 5} more")

    # Human checkpoint
    approved = prompt_human_approval(
        message=f"Review the {format_type} prompts above.\nAre these appropriate for generating {positive_class} vs {negative_class} pairs?",
        require_confirmation=True
    )

    if not approved:
        raise ValueError("User rejected generated prompts")

    # Generate remaining prompts
    remaining = max(0, num_pairs_target * 2 - len(prompts))  # 2x buffer
    if remaining > 0:
        print(f"\nGenerating {remaining} more prompts...")
        for i in range(remaining):
            try:
                if format_type == "qa":
                    prompt = llm_agent.generate_question(
                        positive_class=positive_class,
                        negative_class=negative_class,
                        topic=None
                    )
                else:
                    prompt = llm_agent.generate_statement(
                        positive_class=positive_class,
                        negative_class=negative_class,
                        is_positive=True,
                        topic=None
                    )
                prompts.append(prompt)
            except Exception as e:
                print(f"Warning: Failed to generate prompt: {e}")

        print(f"Total prompts generated: {len(prompts)}")

    return prompts


# ==============================================================================
# PHASE 4: CONTRASTIVE PAIR GENERATION (ON-POLICY)
# ==============================================================================

def phase4_contrastive_pair_generation(
    config: Dict,
    on_policy_generator: OnPolicyGenerator,
    quality_validator: QualityValidator,
    llm_agent: LLMAgent,
    prompts: List[str],
    num_pairs_target: int
) -> List[ContrastivePair]:
    """
    Generate contrastive pairs using on-policy steering.

    Returns:
        List of validated ContrastivePair objects
    """
    print("\n" + "="*60)
    print("PHASE 4: CONTRASTIVE PAIR GENERATION (ON-POLICY)")
    print("="*60)

    format_type = config['generation']['formats'][0] if config['generation']['formats'] else "qa"
    checkpoint_frequency = config['validation']['checkpoint_frequency']
    require_llm_approval = config['validation']['require_llm_approval']
    require_threshold = config['validation']['require_activation_threshold']
    min_cosine_distance = config['validation']['min_cosine_distance']

    positive_class = config['concept']['positive_class']
    negative_class = config['concept']['negative_class']
    concept = positive_class

    valid_pairs = []
    prompt_idx = 0

    while len(valid_pairs) < num_pairs_target and prompt_idx < len(prompts):
        prompt = prompts[prompt_idx]
        prompt_idx += 1

        display_progress(len(valid_pairs), num_pairs_target, "Valid pairs collected")

        print(f"\n--- Generating pair {len(valid_pairs) + 1} ---")
        print(f"Prompt: {prompt[:80]}...")

        # Generate pair
        try:
            pair = on_policy_generator.generate_steered_pair(prompt, format_type)
        except Exception as e:
            print(f"Failed to generate pair: {e}")
            continue

        if pair is None:
            print("Pair generation returned None, skipping...")
            continue

        # Validate pair
        is_valid = True
        validation_details = {}

        # Check activation threshold
        if require_threshold:
            passes_threshold, cosine_distance = quality_validator.check_activation_threshold(
                pair, min_cosine_distance
            )
            validation_details['passes_threshold'] = passes_threshold
            validation_details['cosine_distance'] = cosine_distance

            if not passes_threshold:
                print(f"  ✗ Failed activation threshold: {cosine_distance:.4f} < {min_cosine_distance}")
                is_valid = False

        # LLM judgment
        if require_llm_approval and is_valid:
            try:
                llm_approved, llm_details = quality_validator.validate_with_llm_judge(
                    pair, llm_agent, concept, positive_class, negative_class
                )
                validation_details['llm_approved'] = llm_approved
                validation_details['llm_details'] = llm_details

                if not llm_approved:
                    print(f"  ✗ Failed LLM judgment")
                    is_valid = False
            except Exception as e:
                print(f"  Warning: LLM judgment failed: {e}")
                is_valid = False

        if is_valid:
            valid_pairs.append(pair)
            print(f"  ✓ Valid pair ({len(valid_pairs)}/{num_pairs_target})")
        else:
            print(f"  ✗ Invalid pair, continuing...")

        # Human checkpoint
        if len(valid_pairs) % checkpoint_frequency == 0 and len(valid_pairs) > 0:
            # Display recent pairs
            recent_pairs = valid_pairs[-checkpoint_frequency:]
            display_data = []
            for p in recent_pairs:
                display_data.append({
                    'prompt': p.prompt if hasattr(p, 'prompt') else "N/A",
                    'positive': p.positive_class if hasattr(p, 'positive_class') else p.positive_text,
                    'negative': p.negative_class if hasattr(p, 'negative_class') else p.negative_text,
                    'valid': True
                })
            display_contrastive_pairs(display_data, len(valid_pairs) - checkpoint_frequency)

            approved = prompt_human_approval(
                message=f"Collected {len(valid_pairs)}/{num_pairs_target} valid pairs.\nReview the recent pairs above.",
                require_confirmation=True
            )

            if not approved:
                raise ValueError("User stopped pair generation")

    if len(valid_pairs) < num_pairs_target:
        print(f"\nWarning: Only collected {len(valid_pairs)}/{num_pairs_target} pairs (ran out of prompts)")

    print(f"\n✓ Collected {len(valid_pairs)} valid contrastive pairs")
    return valid_pairs


# ==============================================================================
# PHASE 5: PROBE TRAINING
# ==============================================================================

def phase5_probe_training(
    config: Dict,
    pairs: List[ContrastivePair]
) -> tuple:
    """
    Train logistic regression probe.

    Returns:
        Tuple of (ProbeResults, probe_direction)
    """
    print("\n" + "="*60)
    print("PHASE 5: PROBE TRAINING")
    print("="*60)

    probe_config = config['probe']

    trainer = ProbeTrainer(
        solver=probe_config['solver'],
        max_iter=probe_config['max_iter'],
        C=probe_config['C'],
        random_state=probe_config['random_state']
    )

    print(f"\nTraining logistic regression probe on {len(pairs)} pairs...")
    results = trainer.train_from_pairs(
        pairs=pairs,
        test_size=1.0 - probe_config['train_test_split'],
        balance_classes=probe_config['balance_classes']
    )

    print("\n" + str(results))

    return results, results.probe_direction


# ==============================================================================
# PHASE 6: ANALYSIS & COMPARISON
# ==============================================================================

def phase6_analysis(
    config: Dict,
    sae_manager: SAEManager,
    neuronpedia_client: NeuronpediaClient,
    probe_direction: np.ndarray,
    steering_feature_index: int,
    probe_results: Any
) -> Dict[str, Any]:
    """
    Analyze probe alignment with SAE features.

    Returns:
        Analysis results dictionary
    """
    print("\n" + "="*60)
    print("PHASE 6: ANALYSIS & COMPARISON")
    print("="*60)

    print("\nAnalyzing probe alignment with SAE features...")
    analysis = sae_manager.analyze_probe_alignment(
        probe_direction=probe_direction,
        steering_feature_index=steering_feature_index,
        top_k=20,
        neuronpedia_client=neuronpedia_client
    )

    print(f"\n✓ Probe vs Steering Feature Similarity: {analysis['probe_steering_similarity']:.4f}")
    print(f"  Steering Feature: {analysis['steering_feature_label']}")

    print("\nTop 10 SAE Features Aligned with Probe:")
    for i, feat in enumerate(analysis['top_aligned_features'][:10], 1):
        marker = "🎯" if feat['is_steering_feature'] else "  "
        print(f"  {marker} {i}. [{feat['index']}] {feat['label'][:50]} (sim: {feat['similarity']:.4f})")

    # Display summary
    top_features_for_display = [
        {
            'index': f['index'],
            'label': f['label'],
            'similarity': f['similarity']
        }
        for f in analysis['top_aligned_features'][:10]
    ]

    display_probe_results(
        accuracy=probe_results.test_accuracy,
        auc=probe_results.test_auc,
        train_size=probe_results.train_size,
        test_size=probe_results.test_size,
        top_features=top_features_for_display
    )

    return analysis


# ==============================================================================
# SAVE RESULTS
# ==============================================================================

def save_results(
    output_dir: Path,
    config: Dict,
    pairs: List[ContrastivePair],
    probe_results: Any,
    probe_direction: np.ndarray,
    analysis: Dict[str, Any],
    steering_feature_index: int,
    steering_feature_label: str,
    final_concept: str,
    amplification: float
):
    """Save all pipeline results."""
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    # Save config
    save_yaml(config, output_dir / "config.yaml")
    print(f"✓ Saved config to: {output_dir / 'config.yaml'}")

    # Save pairs
    pairs_data = [pair.to_dict() for pair in pairs]
    with open(output_dir / "pairs.json", 'w') as f:
        json.dump(pairs_data, f, indent=2)
    print(f"✓ Saved pairs to: {output_dir / 'pairs.json'}")

    # Save activations
    activations_dir = output_dir / "activations"
    activations_dir.mkdir(exist_ok=True)
    for i, pair in enumerate(pairs):
        if pair.positive_activations is not None:
            np.save(activations_dir / f"pair_{i}_positive.npy", pair.positive_activations)
            np.save(activations_dir / f"pair_{i}_negative.npy", pair.negative_activations)
    print(f"✓ Saved activations to: {activations_dir}")

    # Save probe
    probe_dir = output_dir / "probe"
    probe_dir.mkdir(exist_ok=True)
    np.save(probe_dir / "probe_direction.npy", probe_direction)
    with open(probe_dir / "probe_results.json", 'w') as f:
        json.dump(probe_results.to_dict(), f, indent=2)
    print(f"✓ Saved probe to: {probe_dir}")

    # Save analysis
    with open(output_dir / "analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"✓ Saved analysis to: {output_dir / 'analysis.json'}")

    # Save pipeline metadata
    metadata = {
        'steering_feature_index': steering_feature_index,
        'steering_feature_label': steering_feature_label,
        'final_concept': final_concept,
        'amplification': amplification,
        'num_pairs': len(pairs),
        'probe_accuracy': probe_results.test_accuracy,
        'probe_auc': probe_results.test_auc,
        'probe_steering_similarity': analysis['probe_steering_similarity']
    }
    with open(output_dir / "pipeline_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved pipeline metadata to: {output_dir / 'pipeline_metadata.json'}")

    # Save human-readable summary
    with open(output_dir / "summary.txt", 'w') as f:
        f.write("="*60 + "\n")
        f.write("SAE-GUIDED PROBE CREATION PIPELINE SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Concept: {config['concept']['positive_class']} vs {config['concept']['negative_class']}\n")
        f.write(f"Final Concept Used: {final_concept}\n\n")
        f.write(f"Steering Feature: [{steering_feature_index}] {steering_feature_label}\n")
        f.write(f"Amplification: {amplification}\n\n")
        f.write(f"Pairs Generated: {len(pairs)}\n")
        f.write(f"Train/Test Split: {probe_results.train_size}/{probe_results.test_size}\n\n")
        f.write("PROBE PERFORMANCE:\n")
        f.write(f"  Test Accuracy: {probe_results.test_accuracy:.4f}\n")
        f.write(f"  Test AUC-ROC: {probe_results.test_auc:.4f}\n")
        f.write(f"  Test Precision: {probe_results.test_precision:.4f}\n")
        f.write(f"  Test Recall: {probe_results.test_recall:.4f}\n")
        f.write(f"  Test F1: {probe_results.test_f1:.4f}\n\n")
        f.write("ALIGNMENT ANALYSIS:\n")
        f.write(f"  Probe vs Steering Feature: {analysis['probe_steering_similarity']:.4f}\n\n")
        f.write("Top 10 SAE Features Aligned with Probe:\n")
        for i, feat in enumerate(analysis['top_aligned_features'][:10], 1):
            marker = "*" if feat['is_steering_feature'] else " "
            f.write(f" {marker} {i}. [{feat['index']}] {feat['label'][:60]} (sim: {feat['similarity']:.4f})\n")
    print(f"✓ Saved summary to: {output_dir / 'summary.txt'}")

    print(f"\n✓ All results saved to: {output_dir}")


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main(config_path: str = "config.yaml"):
    """Run the SAE-guided probe creation pipeline."""
    print("="*60)
    print("SAE-GUIDED PROBE CREATION PIPELINE")
    print("="*60)

    # Load configuration
    print("\nLoading configuration...")
    config = load_config(config_path)
    env_vars = load_env_vars()

    # Validate API keys
    if not env_vars.get('openrouter_api_key'):
        raise ValueError("OPENROUTER_API_KEY not found in .env file")

    # Create output directory
    output_dir = create_output_dir(
        config['output']['save_dir'],
        config['output']['timestamp_format']
    )
    print(f"Output directory: {output_dir}")

    # Initialize components
    print("\n" + "="*60)
    print("INITIALIZING COMPONENTS")
    print("="*60)

    print("\n1. LLM Agent (OpenRouter)...")
    llm_agent = LLMAgent(
        api_key=env_vars['openrouter_api_key'],
        model=config['llm_agent']['model'],
        temperature=config['llm_agent']['temperature'],
        max_tokens=config['llm_agent']['max_tokens'],
        api_base=config['llm_agent']['api_base']
    )

    print("\n2. Model Manager (Llama)...")
    model_manager = ModelManager(
        model_name=config['model']['name'],
        device=config['model']['device'],
        load_in_8bit=config['model']['load_in_8bit'],
        max_length=config['model']['max_length']
    )

    print("\n3. SAE Manager...")
    sae_manager = SAEManager(
        repo_id=config['sae']['repo_id'],
        filename=config['sae'].get('filename'),
        cache_dir=config['sae']['cache_dir'],
        device='cpu'  # Will load to GPU later if needed
    )
    sae_manager.load_sae()

    print("\n4. Neuronpedia Client...")
    api_key = os.getenv(config['neuronpedia']['api_key_env'])
    if not api_key:
        raise ValueError(f"{config['neuronpedia']['api_key_env']} not found")
    neuronpedia_client = NeuronpediaClient(
        api_key=api_key,
        model_id=config['neuronpedia']['model_id'],
        layer_id=config['neuronpedia']['layer_id'],
        cache_dir=config['neuronpedia'].get('cache_dir')
    )

    print("\n5. Quality Validator...")
    goodfire_client = None  # Optional for new pipeline
    quality_validator = QualityValidator(
        goodfire_client=goodfire_client,
        positive_queries=config['validation']['positive_feature_queries'],
        negative_queries=config['validation']['negative_feature_queries'],
        top_k=10,
        sae_manager=sae_manager,
        sae_top_k=config['sae']['top_k_features'],
        sae_aggregation=config['sae']['aggregation'],
        neuronpedia_client=neuronpedia_client
    )

    # Run pipeline phases
    try:
        # Phase 1: SAE Feature Search & Approval
        steering_feature_index, steering_feature_label, decoder_direction, final_concept = \
            phase1_sae_feature_search_approval(config, llm_agent, neuronpedia_client, sae_manager)

        # Phase 2: Amplification Tuning
        amplification = phase2_amplification_tuning(
            config, llm_agent, model_manager, decoder_direction, config['model']['layer'], final_concept
        )

        # Create on-policy generator
        steering_config = SteeringConfig(
            vector=decoder_direction,
            layer=config['model']['layer'],
            amplification=amplification
        )
        on_policy_generator = OnPolicyGenerator(
            model_manager=model_manager,
            steering_config=steering_config,
            max_new_tokens=config['llm_agent']['max_tokens'],
            temperature=config['llm_agent']['temperature'],
            exclude_last_n_tokens=config['generation']['apollo']['exclude_last_n_tokens'],
            min_assistant_tokens=config['generation']['apollo']['min_assistant_tokens']
        )

        # Phase 3: Question Generation
        prompts = phase3_question_generation(
            config, llm_agent, config['generation']['num_pairs']
        )

        # Phase 4: Contrastive Pair Generation
        pairs = phase4_contrastive_pair_generation(
            config, on_policy_generator, quality_validator, llm_agent,
            prompts, config['generation']['num_pairs']
        )

        # Phase 5: Probe Training
        probe_results, probe_direction = phase5_probe_training(config, pairs)

        # Phase 6: Analysis
        analysis = phase6_analysis(
            config, sae_manager, neuronpedia_client, probe_direction,
            steering_feature_index, probe_results
        )

        # Save results
        save_results(
            output_dir, config, pairs, probe_results, probe_direction, analysis,
            steering_feature_index, steering_feature_label, final_concept, amplification
        )

        print("\n" + "="*60)
        print("PIPELINE COMPLETE!")
        print("="*60)
        print(f"\nResults saved to: {output_dir}")
        print(f"\nProbe Performance:")
        print(f"  Accuracy: {probe_results.test_accuracy:.4f}")
        print(f"  AUC-ROC: {probe_results.test_auc:.4f}")
        print(f"  Probe-Steering Similarity: {analysis['probe_steering_similarity']:.4f}")

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAE-guided probe creation pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    args = parser.parse_args()
    main(args.config)
