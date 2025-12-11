# Int vs Prod AIS Data Comparison Tools

This directory contains tools for comparing Atlas activity classification outputs between the **integration (int)** and **production (prod)** environments, which use different AIS data providers:

- **Integration**: Uses **Orbcomm** AIS data
- **Production**: Uses **Spire** AIS data

## Purpose

When the same vessel subpath is classified differently between int and prod, it's often due to underlying differences in the AIS data from the two providers. These tools help:

1. **Identify classification discrepancies** between environments
2. **Analyze distributional differences** in the underlying trajectory data (SOG, COG, message density, time gaps)
3. **Understand how data provider differences** impact model behavior

## Scripts

### `compare_int_prod_models.py`

Fetches trajectory data and runs classification for paired subpaths from int and prod, then compares the outputs.

**Setup (port forwards required):**
```bash
# Int endpoints
kubectl --context=gke_skylight-int-a-r2d2_us-west1-a_sky-int-a port-forward service/subpath-api 5102:5102 &
kubectl --context=gke_skylight-int-a-r2d2_us-west1-a_sky-int-a port-forward deployment/subpath-activity-classification-worker-and-atlas 8080:8080 &

# Prod endpoints (different ports)
kubectl --context=gke_skylight-prod-a-a72a_us-west1-a_sky-prod-a port-forward service/subpath-api 5103:5102 &
kubectl --context=gke_skylight-prod-a-a72a_us-west1-a_sky-prod-a port-forward deployment/subpath-activity-classification-worker-and-atlas 8081:8080 &
```

**Usage:**
```bash
python compare_int_prod_models.py --csv subpath_pairs.csv --output-dir results/
```

**Input CSV format:**
```
int_subpath_id,prod_subpath_id
subpath_123,subpath_456
subpath_789,subpath_012
```

**Output artifacts:**
- `comparison_statistics.csv` - Per-subpath statistics and classification results
- `confusion_matrix_postprocessing.png` - Confusion matrix of post-processed outputs
- `confusion_matrix_preprocessing.png` - Confusion matrix of pre-processed outputs

### `analyze_differences.py`

Analyzes the output from `compare_int_prod_models.py` to identify patterns in classification differences and their relationship to underlying data distributions.

**Usage:**
```bash
python analyze_differences.py comparison_statistics.csv -o output_dir/ --plot
```

**Output artifacts:**
- `classification_summary.txt` - Summary of classification agreement/disagreement
- `classification_differences.csv` - Rows where int and prod classifications differ
- `error_transitions.csv` - Counts of each int→prod classification transition
- `per_transition_statistics.csv` - Distributional statistics grouped by transition type
- `large_magnitude_differences.csv` - Metrics with largest int-prod differences
- `distribution_plots/` - Visualizations comparing int vs prod distributions by transition

## Key Metrics Compared

| Metric | Description |
|--------|-------------|
| `sog_mean/min/max/std` | Speed over ground statistics |
| `cog_mean/min/max/std` | Course over ground statistics |
| `num_messages` | Number of AIS messages in the track |
| `avg_time_gap_seconds` | Average time between consecutive messages |
| `overlap_percentage_of_union` | Timestamp overlap between int and prod tracks |

## Example Workflow

1. Generate a CSV of subpath pairs to compare (e.g., from event matching)
2. Run the comparison:
   ```bash
   python compare_int_prod_models.py --csv pairs.csv --output-dir results/ --save-tracks
   ```
3. Analyze the results:
   ```bash
   python analyze_differences.py results/comparison_statistics.csv -o results/ --plot
   ```
4. Review distribution plots to understand which data characteristics drive classification differences

## Interpreting Results

When examining transitions like `transiting -> fishing` (int classifies as transiting, prod as fishing):

- **Higher message density** in prod data may provide more evidence for fishing patterns
- **Lower SOG variance** in int data may cause transiting classification
- **Different timestamp coverage** can capture different portions of vessel behavior

The distribution plots and per-transition statistics help identify which data characteristics are systematically different between providers and how they correlate with classification disagreements.
