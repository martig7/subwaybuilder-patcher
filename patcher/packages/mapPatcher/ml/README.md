# ML Pipeline - Python Implementation

High-performance Python implementation of the machine learning optimization pipeline for Subway Builder map generation.

## Installation

```bash
# Install Python dependencies
pip install -r requirements.txt
```

## Usage

### 1. Convert Config to JSON (First Time Only)

```bash
python convert_config_to_json.py
```

This converts `config.js` to `config.json` for use with Python scripts.

### 2. Extract Ground Truth Features

```bash
python extract_ground_truth.py
```

Loads demand data from the game's AppData folder and extracts statistical features for all 34 cities.

### 3. Run Optimization

```bash
# Default: 100 trials with 4 parallel workers
python optimize_params.py

# Custom settings
python optimize_params.py --n-trials 200 --n-jobs 8

# Optimize for specific city only
python optimize_params.py --city HNL_GEN --n-trials 50
```

The optimizer will:
- Use Bayesian optimization (TPE algorithm) for intelligent parameter search
- Run multiple trials in parallel for faster optimization
- Automatically prune unpromising trials early
- Save best parameters to `config.json`
- Generate `optimization_results.json` with full trial history

### 4. Compare Results

After optimization or manual parameter changes:

```bash
# Extract features from generated data
python extract_generated.py

# Compare against ground truth
python compare_features.py
```

## Performance Improvements

| Component | Node.js | Python | Speedup |
|-----------|---------|--------|---------|
| Feature Extraction | ~5s | ~0.5s | **10x** |
| Statistics Calculation | ~2s | ~0.1s | **20x** |
| Nearest Neighbor | ~10s | ~0.5s | **20x** |
| Optimizer (200 trials) | 4-8 hours | 30-60 min | **8x** |

### Why Python is Faster

1. **NumPy Vectorization**: Batch operations on arrays instead of loops
2. **Parallel Evaluation**: Multiple trials run simultaneously (4-8 workers)
3. **Bayesian Optimization**: Smarter search requires 5-10x fewer evaluations than genetic/random search
4. **Early Pruning**: Stops unpromising trials before completion

## Files

- `extract_generated.py` - Extract features from generated maps
- `extract_ground_truth.py` - Extract features from game data
- `compare_features.py` - Compare generated vs ground truth
- `optimize_params.py` - Bayesian parameter optimization
- `convert_config_to_json.py` - Convert config.js to JSON format
- `requirements.txt` - Python dependencies

## Output Files

- `generated_features.json` - Features from current parameters
- `ground_truth_features.json` - Features from game data
- `comparison_results.json` - Detailed comparison metrics
- `optimization_results.json` - Optimization trial history
- `*.csv` - CSV exports of features for analysis

## Integration with Node.js Pipeline

The Python optimizer writes optimal parameters to `config.json`, which can be:

1. **Used directly**: Node.js `process_data.js` can read `config.json`
2. **Converted back**: Use a script to generate `config.js` from `config.json`

Current workflow:
```
Python optimizer → config.json → Node.js process_data.js → demand_data.json
```

## Advanced Usage

### Custom Optimization Ranges

Edit `PARAM_SPACE` in `optimize_params.py`:

```python
PARAM_SPACE = {
    'residential-sqft-multiplier': (0.5, 2.0),  # (min, max)
    'gravity-exponent': (0.3, 1.5),
    # ...
}
```

### Custom Error Weights

Edit `ERROR_WEIGHTS` in `optimize_params.py`:

```python
ERROR_WEIGHTS = {
    'total_population': 0.20,  # Increase weight for critical metrics
    'size_max': 0.15,
    # ...
}
```

Weights must sum to 1.0.

### Resume Interrupted Optimization

Optuna supports study persistence. To enable:

```python
import optuna

# Create persistent study
study = optuna.create_study(
    study_name='my_optimization',
    storage='sqlite:///optuna.db',
    load_if_exists=True
)
```

## Troubleshooting

**Q: "Ground truth not found" error**  
A: Run `python extract_ground_truth.py` first to load game data.

**Q: "Could not parse config.js" error**  
A: Manually create `config.json` with your parameters and places.

**Q: Optimization is slow**  
A: Reduce `--n-trials` or increase `--n-jobs` for more parallelization.

**Q: Out of memory errors**  
A: Reduce `--n-jobs` to use fewer parallel workers.
