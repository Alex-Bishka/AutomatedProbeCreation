# SAE Probe Data Results Viewer

Flask-based web interface for visualizing SAE-guided probe data generation results.

## Features

- **Run Browser**: View all experimental runs with quality metrics
- **Detailed Analysis**: Inspect individual pairs with SAE feature activations
- **Token-Level Heatmaps**: Visualize activation patterns following Apollo Research methodology
- **Scientific Styling**: Clean, professional Nature/Science journal-inspired design

## Installation

Flask and Plotly are already included in the main project dependencies. No additional installation needed!

## Usage

### Start the viewer:

**From the project root** (not from the viewer directory):

```bash
cd /mnt/SharedData/Projects/AutomatedProbeCreation/sae-usage
uv run python outputs/viewer/app.py
```

The viewer will automatically use the project's virtual environment with all dependencies.

Then open in your browser:
```
http://localhost:5000
```

## What You'll See

### Home Page
- List of all runs sorted by timestamp
- Number of pairs per run
- Average quality scores
- Validation status

### Run Detail Page
- Quality metrics dashboard (avg quality, feature overlap)
- Pair-by-pair analysis:
  - Positive and negative example text
  - Top-10 SAE features with activation strengths
  - Token-level activation heatmaps
  - Quality scores per pair

## Data Sources

The viewer automatically reads from:
- `../[timestamp]/pairs.json` - Generated contrastive pairs
- `../[timestamp]/validation_results.json` - SAE validation results
- `../[timestamp]/config.yaml` - Run configuration
- `../[timestamp]/activations/*.npy` - Token-level activations

## API Endpoints

- `GET /` - Home page (run browser)
- `GET /run/<timestamp>` - View specific run details
- `GET /api/activation/<timestamp>/<pair_id>/<label>` - Get activation data for heatmaps

## Customization

Edit `static/style.css` to customize the appearance.

The color scheme follows Nature/Science journal standards:
- Primary: `#0066cc`
- Success: `#2e7d32`
- Warning: `#ed6c02`
- Error: `#d32f2f`

## Troubleshooting

**No runs showing up?**
- Make sure you've run the pipeline at least once: `uv run python main.py`
- Check that output directories exist in `../`

**Heatmaps not loading?**
- Ensure activation files exist in `../[timestamp]/activations/`
- Check browser console for errors

**Port already in use?**
- Change port: `flask run --port 5001`
