# Apollo baseline (Llama 3.1 8B, all layers)

This folder mirrors the Apollo probe setup from `deception-detection`, but targets
`llama-3.1-8b-instruct` and uses all layers for probe training.

Run the baseline (uses max pooling by default):

```bash
python apollo_baseline/run_apollo_baseline.py run --config apollo_baseline/configs/apollo_llama3_1_8b_all_layers.yaml
```

Outputs land under `apollo_baseline/results/<experiment_name>`.

Notes:
- Layer indices follow the `deception_detection` hidden-state indexing (0-based),
  so Llama 3.1 8B uses layers 0-31.
- On-policy evaluations require the model weights to be available locally.
- The baseline uses max pooling over token-level scores for sample-level comparison.
- Multi-GPU runs use `device_map="auto"` and will split across visible GPUs.
- Optional: set `APOLLO_MAX_GPU_MEM_GB` to cap per-GPU memory (e.g. `APOLLO_MAX_GPU_MEM_GB=22`).
- Detection mask HTML generation is skipped by default to avoid dataset formatting assertions.
  Set `APOLLO_SKIP_DETECTION_MASKS=0` if you want to generate `detection_masks.html`.
