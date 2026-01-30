# ML Optimization Pipeline

Bayesian optimization of map generation parameters using Optuna.

## Setup

```bash
pip install -r requirements.txt
```

## Workflow

### 1. Extract Ground Truth

```bash
python extract_ground_truth.py
```

Extracts statistical features from the game's original demand data for all cities.

### 2. Run Optimization

```bash
python optimize_params.py                           # 100 trials, 4 workers
python optimize_params.py --n-trials 200 --n-jobs 8 # Custom
python optimize_params.py --city LON_GEN            # Single city
```

Outputs:
- `../config.json` — Best parameters (used by Node.js pipeline)
- `optimization_results.json` — Full trial history

### 3. Compare Results (Optional)

```bash
python extract_generated.py   # Extract features from generated data
python compare_features.py    # Compare against ground truth
```

## Key Files

| File | Purpose |
|------|---------|
| `process_data.py` | Core data processing (buildings, neighborhoods, connections) |
| `optimize_params.py` | Bayesian optimization with TPE sampler |
| `extract_ground_truth.py` | Load original game data features |
| `extract_generated.py` | Load generated data features |
| `compare_features.py` | Feature comparison and error calculation |

## Customization

**Parameter ranges** — Edit `PARAM_SPACE` in `optimize_params.py`

**Error weights** — Edit `ERROR_WEIGHTS` in `optimize_params.py` (must sum to 1.0)

## Troubleshooting

| Error | Solution |
|-------|----------|
| "Ground truth not found" | Run `python extract_ground_truth.py` first |
| Out of memory | Reduce `--n-jobs` |
| Slow optimization | Increase `--n-jobs` or reduce `--n-trials` |
