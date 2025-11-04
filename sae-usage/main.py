#!/usr/bin/env python3
"""Main pipeline for SAE-guided automated probe data generation."""

import argparse
import json
import numpy as np
from pathlib import Path
from src import (
    load_config,
    load_env_vars,
    create_output_dir,
    LLMAgent,
    ModelManager,
    GoodfireClient,
    DataGenerator,
    QualityValidator,
    save_yaml
)


def save_results(output_dir: Path, pairs, validation_results, config):
    """Save all results to output directory.

    Args:
        output_dir: Output directory path
        pairs: List of ContrastivePair objects
        validation_results: List of ValidationResult objects
        config: Configuration dictionary
    """
    print(f"\n{'='*60}")
    print("Saving results...")
    print(f"{'='*60}")

    # Save configuration
    save_yaml(config, output_dir / "config.yaml")
    print(f"Saved config to: {output_dir / 'config.yaml'}")

    # Save pairs metadata
    pairs_data = [pair.to_dict() for pair in pairs]
    with open(output_dir / "pairs.json", 'w') as f:
        json.dump(pairs_data, f, indent=2)
    print(f"Saved pairs metadata to: {output_dir / 'pairs.json'}")

    # Save validation results
    validation_data = [result.to_dict() for result in validation_results]
    with open(output_dir / "validation_results.json", 'w') as f:
        json.dump(validation_data, f, indent=2)
    print(f"Saved validation results to: {output_dir / 'validation_results.json'}")

    # Save activations (for pairs that have them)
    activations_dir = output_dir / "activations"
    activations_dir.mkdir(exist_ok=True)

    for i, pair in enumerate(pairs):
        if pair.positive_activations is not None:
            np.save(
                activations_dir / f"pair_{i}_positive.npy",
                pair.positive_activations
            )
            np.save(
                activations_dir / f"pair_{i}_negative.npy",
                pair.negative_activations
            )
    print(f"Saved activations to: {activations_dir}")

    # Save text outputs for easy review
    with open(output_dir / "pairs_readable.txt", 'w') as f:
        for i, pair in enumerate(pairs):
            f.write(f"{'='*60}\n")
            f.write(f"PAIR {i} ({pair.format_type} format)\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"POSITIVE ({config['concept']['positive_class']}):\n")
            f.write(f"{pair.positive_text}\n\n")
            f.write(f"NEGATIVE ({config['concept']['negative_class']}):\n")
            f.write(f"{pair.negative_text}\n\n")
            if validation_results:
                f.write(f"Quality Score: {validation_results[i].quality_score:.2f}\n")
            f.write("\n\n")
    print(f"Saved readable text to: {output_dir / 'pairs_readable.txt'}")

    print(f"\nAll results saved to: {output_dir}")


def main(config_path: str = "config.yaml"):
    """Run the main pipeline.

    Args:
        config_path: Path to configuration file
    """
    print("="*60)
    print("SAE-GUIDED AUTOMATED PROBE DATA GENERATION")
    print("="*60)

    # Load configuration
    print("\nLoading configuration...")
    config = load_config(config_path)
    env_vars = load_env_vars()

    # Validate API keys
    if not env_vars['goodfire_api_key']:
        raise ValueError("GOODFIRE_API_KEY not found in .env file")
    if not env_vars['openrouter_api_key']:
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

    print("\n1. Initializing LLM Agent (OpenRouter)...")
    llm_agent = LLMAgent(
        api_key=env_vars['openrouter_api_key'],
        model=config['llm_agent']['model'],
        temperature=config['llm_agent']['temperature'],
        max_tokens=config['llm_agent']['max_tokens'],
        api_base=config['llm_agent']['api_base']
    )

    print("\n2. Initializing Model Manager (Llama)...")
    model_manager = ModelManager(
        model_name=config['model']['name'],
        device=config['model']['device'],
        load_in_8bit=config['model']['load_in_8bit'],
        max_length=config['model']['max_length']
    )

    print("\n3. Initializing Goodfire Client...")
    goodfire_client = GoodfireClient(
        api_key=env_vars['goodfire_api_key'],
        model_variant=config['goodfire']['model_variant'],
        layer=config['goodfire']['layer']
    )

    print("\n4. Initializing Data Generator...")
    data_generator = DataGenerator(
        llm_agent=llm_agent,
        model_manager=model_manager,
        positive_class=config['concept']['positive_class'],
        negative_class=config['concept']['negative_class'],
        layer=config['model']['layer'],
        exclude_last_n=config['generation']['apollo']['exclude_last_n_tokens']
    )

    print("\n5. Initializing Quality Validator...")
    quality_validator = QualityValidator(
        goodfire_client=goodfire_client,
        positive_queries=config['validation']['positive_feature_queries'],
        negative_queries=config['validation']['negative_feature_queries'],
        top_k=config['goodfire']['top_k_features']
    )

    # Generate contrastive pairs
    print("\n" + "="*60)
    print("GENERATING CONTRASTIVE PAIRS")
    print("="*60)

    pairs = data_generator.generate_pairs(
        num_pairs=config['generation']['num_pairs'],
        formats=config['generation']['formats'],
        generate_topics=True
    )

    print(f"\n✓ Successfully generated {len(pairs)} contrastive pairs")

    # Validate pairs
    print("\n" + "="*60)
    print("VALIDATING PAIRS WITH SAE FEATURES")
    print("="*60)

    validation_results = quality_validator.validate_pairs(pairs)

    print(f"\n✓ Successfully validated {len(pairs)} pairs")

    # Save results
    save_results(output_dir, pairs, validation_results, config)

    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Review pairs in pairs_readable.txt")
    print("2. Check validation results in validation_results.json")
    print("3. Use activations in activations/ for probe training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAE-guided automated probe data generation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )

    args = parser.parse_args()
    main(args.config)
