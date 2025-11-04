# SAE-Guided Automated Probe Data Generation

Automated system for generating high-quality contrastive data pairs for probe training, using Sparse Autoencoder (SAE) features from Goodfire API to validate data quality.

## Overview

This project implements the methodology from Apollo Research's "Detecting Strategic Deception Using Linear Probes" paper, combined with SAE-based quality validation using Goodfire's interpretability API.

### Key Features

- **Automated Data Generation**: Generate contrastive pairs for any concept (deception, toxicity, bias, etc.)
- **On-Policy Responses**: Uses the target model (Llama 3.1 8B) to generate responses
- **SAE-Based Validation**: Leverages interpretable SAE features to validate data quality
- **Apollo Methodology**: Extracts token-level activations following best practices
- **Flexible Configuration**: Easy to adapt to different models and concepts

## Architecture

```
┌─────────────────┐
│  LLM Agent      │  Generate prompts/questions via OpenRouter
│  (OpenRouter)   │
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Model Manager  │  Generate on-policy responses & extract activations
│  (Llama 3.1 8B) │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Quality         │  Validate using SAE features from Goodfire API
│ Validator       │
└─────────────────┘
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for Llama 3.1 8B)
- API Keys:
  - Goodfire API key
  - OpenRouter API key
  - Hugging Face token

### Setup

1. Clone and navigate to the project:
```bash
cd sae-usage
```

2. Install dependencies using `uv`:
```bash
uv sync
```

3. Set up your `.env` file with API keys:
```bash
GOODFIRE_API_KEY=your_goodfire_key
OPENROUTER_API_KEY=your_openrouter_key
HF_TOKEN=your_huggingface_token
```

## Usage

### Basic Usage

Run the pipeline with default configuration:
```bash
uv run python main.py
```

### Custom Configuration

Modify `config.yaml` to customize:
- Target concept (positive/negative classes)
- Model settings (name, layer, device)
- LLM agent model
- Number of pairs to generate
- Data formats (statement, Q&A)

Run with custom config:
```bash
uv run python main.py --config custom_config.yaml
```

### Configuration Example

```yaml
# Change the concept
concept:
  positive_class: "toxic"
  negative_class: "helpful"

# Use different model
model:
  name: "meta-llama/Llama-3.3-70B-Instruct"
  layer: 50

# Generate more pairs
generation:
  num_pairs: 10
```

## Output Structure

After running the pipeline, outputs are saved to `outputs/TIMESTAMP/`:

```
outputs/20250103_143022/
├── config.yaml                    # Configuration used
├── pairs.json                     # Structured pair data
├── pairs_readable.txt             # Human-readable pairs
├── validation_results.json        # SAE validation results
└── activations/                   # Token-level activations (.npy files)
    ├── pair_0_positive.npy
    ├── pair_0_negative.npy
    └── ...
```

## Methodology

### Data Generation

1. **Statement Format**: LLM generates complete statements exhibiting the concept
2. **Q&A Format**:
   - LLM generates questions
   - Llama generates on-policy responses (one positive, one negative)

### Activation Extraction (Apollo Research Methodology)

- Extract from **all assistant tokens except last 5**
- Keep activations at **token-level** (no mean-pooling during extraction)
- Layer 19 for Llama 3.1 8B, Layer 50 for Llama 3.3 70B

### Quality Validation

1. Search Goodfire API for expected SAE features (e.g., "deception", "honesty")
2. Analyze generated pairs to see which features activate
3. Compute quality score based on overlap with expected features
4. Manual review of validation results

## Extending to Other Concepts

To create probes for different concepts:

1. Modify `config.yaml`:
```yaml
concept:
  positive_class: "your_concept"
  negative_class: "opposite_concept"

validation:
  positive_feature_queries:
    - "your_concept"
    - "synonyms"
  negative_feature_queries:
    - "opposite_concept"
    - "synonyms"
```

2. Run the pipeline

Examples:
- **Bias Detection**: positive="biased", negative="unbiased"
- **Toxicity**: positive="toxic", negative="helpful"
- **Uncertainty**: positive="uncertain", negative="confident"

## Project Structure

```
sae-usage/
├── main.py                        # Main pipeline
├── config.yaml                    # Configuration
├── .env                           # API keys
├── src/
│   ├── __init__.py
│   ├── utils.py                   # Config loading, helpers
│   ├── llm_agent.py               # OpenRouter wrapper
│   ├── model_manager.py           # Llama loading & activation extraction
│   ├── goodfire_client.py         # Goodfire API wrapper
│   ├── data_generator.py          # Contrastive pair generation
│   └── quality_validator.py       # SAE validation
└── outputs/                       # Generated data
```

## Future Work

- [ ] Automated quality thresholds and filtering
- [ ] Iterative scaling (1→10→50 pairs)
- [ ] Probe training implementation
- [ ] Support for additional models
- [ ] Batch processing for large datasets

## References

- [Apollo Research: Detecting Strategic Deception Using Linear Probes](https://www.apolloresearch.ai/research/deception-probes)
- [Goodfire API Documentation](https://docs.goodfire.ai/)
- [Goodfire SAE Models on HuggingFace](https://huggingface.co/Goodfire)

## License

MIT
