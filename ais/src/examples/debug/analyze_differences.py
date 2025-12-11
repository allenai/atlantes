#!/usr/bin/env python3
# type: ignore
"""
Quick script to analyze differences between int and prod statistics.
"""
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_transition_distributions(
    df: pd.DataFrame,
    output_dir: Path,
    top_n_transitions: int = 10
) -> None:
    """Generate distribution plots comparing int vs prod for each transition type.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with comparison statistics
    output_dir : Path
        Directory to save plot files
    top_n_transitions : int
        Number of top transitions to plot
    """
    output_dir = Path(output_dir)
    plots_dir = output_dir / "distribution_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Get rows where predictions differ
    if "post_outputs_different" in df.columns:
        diff_mask = df["post_outputs_different"]
    else:
        diff_mask = df["int_post_postprocessing_class"] != df["prod_post_postprocessing_class"]

    diff_df = df[diff_mask].copy()

    if len(diff_df) == 0:
        print("No differing predictions found, skipping plots.")
        return

    # Create transition column
    diff_df["transition"] = (
        diff_df["int_post_postprocessing_class"].astype(str) +
        " -> " +
        diff_df["prod_post_postprocessing_class"].astype(str)
    )

    # Get top transitions
    transition_counts = diff_df["transition"].value_counts()
    top_transitions = transition_counts.head(top_n_transitions).index.tolist()

    # Define metric groups to plot
    metric_groups = {
        "SOG": [
            ("int_sog_mean", "prod_sog_mean", "SOG Mean"),
            ("int_sog_min", "prod_sog_min", "SOG Min"),
            ("int_sog_max", "prod_sog_max", "SOG Max"),
            ("int_sog_std", "prod_sog_std", "SOG Std"),
        ],
        "COG": [
            ("int_cog_mean", "prod_cog_mean", "COG Mean"),
            ("int_cog_min", "prod_cog_min", "COG Min"),
            ("int_cog_max", "prod_cog_max", "COG Max"),
            ("int_cog_std", "prod_cog_std", "COG Std"),
        ],
        "Messages": [
            ("int_num_messages", "prod_num_messages", "Num Messages"),
            ("int_avg_time_gap_seconds", "prod_avg_time_gap_seconds", "Avg Time Gap (s)"),
        ],
    }

    # Print data availability info
    print("\nData availability in differing predictions:")
    for int_col, prod_col, name in [
        ("int_sog_mean", "prod_sog_mean", "SOG Mean"),
        ("int_num_messages", "prod_num_messages", "Num Messages"),
    ]:
        if int_col in diff_df.columns and prod_col in diff_df.columns:
            int_valid = diff_df[int_col].notna().sum()
            prod_valid = diff_df[prod_col].notna().sum()
            print(f"  {name}: Int has {int_valid}/{len(diff_df)} valid, Prod has {prod_valid}/{len(diff_df)} valid")
    print()

    # Generate plots for each transition type
    for transition in top_transitions:
        trans_df = diff_df[diff_df["transition"] == transition]
        n_samples = len(trans_df)

        # Clean transition name for filename
        safe_transition = transition.replace(" -> ", "_to_").replace(" ", "_")

        for group_name, metrics in metric_groups.items():
            n_metrics = len(metrics)
            fig, axes = plt.subplots(n_metrics, 2, figsize=(12, 3 * n_metrics))
            fig.suptitle(f"{transition} (n={n_samples}) - {group_name} Distributions", fontsize=14)

            if n_metrics == 1:
                axes = [axes]

            for idx, (int_col, prod_col, metric_name) in enumerate(metrics):
                ax_hist = axes[idx][0]
                ax_box = axes[idx][1]

                # Check column existence
                int_col_exists = int_col in trans_df.columns
                prod_col_exists = prod_col in trans_df.columns

                if not int_col_exists and not prod_col_exists:
                    ax_hist.text(0.5, 0.5, f'{metric_name}\nNo data available',
                                ha='center', va='center', transform=ax_hist.transAxes)
                    ax_box.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax_box.transAxes)
                    continue

                # Get values, handling missing columns
                int_values = trans_df[int_col].dropna() if int_col_exists else pd.Series([], dtype=float)
                prod_values = trans_df[prod_col].dropna() if prod_col_exists else pd.Series([], dtype=float)

                n_int = len(int_values)
                n_prod = len(prod_values)

                # Left plot: overlaid histograms
                if n_int > 0 or n_prod > 0:
                    all_values = pd.concat([int_values, prod_values])
                    bins = np.histogram_bin_edges(all_values, bins='auto') if len(all_values) > 0 else 10

                    if n_int > 0:
                        ax_hist.hist(int_values, bins=bins, alpha=0.5, label=f'Int (n={n_int})', color='blue')
                        ax_hist.axvline(int_values.mean(), color='blue', linestyle='--', linewidth=2,
                                       label=f'Int μ: {int_values.mean():.2f}')

                    if n_prod > 0:
                        ax_hist.hist(prod_values, bins=bins, alpha=0.5, label=f'Prod (n={n_prod})', color='orange')
                        ax_hist.axvline(prod_values.mean(), color='orange', linestyle='--', linewidth=2,
                                       label=f'Prod μ: {prod_values.mean():.2f}')

                    ax_hist.set_xlabel(metric_name)
                    ax_hist.set_ylabel('Count')
                    ax_hist.set_title(f'{metric_name} - Histogram (Int: {n_int}, Prod: {n_prod})')
                    ax_hist.legend(fontsize=8)
                else:
                    ax_hist.text(0.5, 0.5, f'{metric_name}\nNo valid data',
                                ha='center', va='center', transform=ax_hist.transAxes)

                # Right plot: box plots side by side
                box_data = []
                box_labels = []

                if n_int > 0:
                    box_data.append(int_values.values)
                    box_labels.append(f'Int\n(n={n_int})')

                if n_prod > 0:
                    box_data.append(prod_values.values)
                    box_labels.append(f'Prod\n(n={n_prod})')

                if box_data:
                    bp = ax_box.boxplot(box_data, labels=box_labels, patch_artist=True)
                    colors = ['lightblue', 'lightyellow'][:len(box_data)]
                    for patch, color in zip(bp['boxes'], colors):
                        patch.set_facecolor(color)
                    ax_box.set_ylabel(metric_name)
                    ax_box.set_title(f'{metric_name} - Box Plot')

                    # Add mean annotations
                    for i, (data, label) in enumerate(zip(box_data, box_labels), 1):
                        mean_val = np.mean(data)
                        ax_box.annotate(f'μ={mean_val:.2f}', xy=(i, mean_val), fontsize=8,
                                       xytext=(5, 0), textcoords='offset points')
                else:
                    ax_box.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax_box.transAxes)

            plt.tight_layout()
            plot_file = plots_dir / f"{safe_transition}_{group_name.lower()}.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved {plot_file}")

    # Generate summary plot: all transitions comparison for key metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Key Metrics Comparison Across Top Transitions", fontsize=14)

    key_metrics = [
        ("int_sog_mean", "prod_sog_mean", "SOG Mean", axes[0, 0]),
        ("int_num_messages", "prod_num_messages", "Num Messages", axes[0, 1]),
        ("int_cog_mean", "prod_cog_mean", "COG Mean", axes[1, 0]),
        ("int_avg_time_gap_seconds", "prod_avg_time_gap_seconds", "Avg Time Gap (s)", axes[1, 1]),
    ]

    for int_col, prod_col, metric_name, ax in key_metrics:
        if int_col not in diff_df.columns or prod_col not in diff_df.columns:
            continue

        # Calculate mean diff for each transition
        trans_means = []
        for trans in top_transitions:
            trans_data = diff_df[diff_df["transition"] == trans]
            int_mean = trans_data[int_col].mean()
            prod_mean = trans_data[prod_col].mean()
            trans_means.append({
                "transition": trans,
                "int_mean": int_mean,
                "prod_mean": prod_mean,
                "diff": int_mean - prod_mean,
            })

        trans_means_df = pd.DataFrame(trans_means)

        x = np.arange(len(top_transitions))
        width = 0.35

        ax.bar(x - width/2, trans_means_df["int_mean"], width, label='Int', color='blue', alpha=0.7)
        ax.bar(x + width/2, trans_means_df["prod_mean"], width, label='Prod', color='orange', alpha=0.7)
        ax.set_xlabel('Transition')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} by Transition')
        ax.set_xticks(x)
        ax.set_xticklabels([t[:20] + '...' if len(t) > 20 else t for t in top_transitions], rotation=45, ha='right', fontsize=8)
        ax.legend()

    plt.tight_layout()
    summary_plot = plots_dir / "summary_all_transitions.png"
    plt.savefig(summary_plot, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {summary_plot}")

    print(f"\nAll distribution plots saved to {plots_dir}")


def analyze_classification_differences(df: pd.DataFrame, output_dir: Path | None = None) -> dict:
    """Analyze classification differences between int and prod.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with comparison statistics
    output_dir : Path | None
        Directory to save summary files. If None, no files are saved.

    Returns
    -------
    dict
        Summary of classification differences
    """
    summary: dict = {}

    print("=" * 80)
    print("CLASSIFICATION DIFFERENCE ANALYSIS")
    print("=" * 80)
    print()

    # Overall statistics
    total_rows = len(df)
    post_diff_mask = df["post_outputs_different"] if "post_outputs_different" in df.columns else (
        df["int_post_postprocessing_class"] != df["prod_post_postprocessing_class"]
    )
    pre_diff_mask = df["pre_outputs_different"] if "pre_outputs_different" in df.columns else (
        df["int_pre_postprocessing_class"] != df["prod_pre_postprocessing_class"]
    )

    post_diff_count = post_diff_mask.sum()
    pre_diff_count = pre_diff_mask.sum()

    print(f"Total rows: {total_rows}")
    print(f"Post-processing differences: {post_diff_count} ({100*post_diff_count/total_rows:.1f}%)")
    print(f"Pre-processing differences: {pre_diff_count} ({100*pre_diff_count/total_rows:.1f}%)")
    print()

    summary["total_rows"] = total_rows
    summary["post_diff_count"] = int(post_diff_count)
    summary["post_diff_pct"] = round(100*post_diff_count/total_rows, 2)
    summary["pre_diff_count"] = int(pre_diff_count)
    summary["pre_diff_pct"] = round(100*pre_diff_count/total_rows, 2)

    # Distribution of int labels
    print("--- Int Post-Processing Label Distribution ---")
    int_post_dist = df["int_post_postprocessing_class"].value_counts(dropna=False)
    print(int_post_dist.to_string())
    summary["int_post_distribution"] = int_post_dist.to_dict()
    print()

    # Distribution of prod labels
    print("--- Prod Post-Processing Label Distribution ---")
    prod_post_dist = df["prod_post_postprocessing_class"].value_counts(dropna=False)
    print(prod_post_dist.to_string())
    summary["prod_post_distribution"] = prod_post_dist.to_dict()
    print()

    # Error type analysis (when predictions differ)
    print("--- Error Types (Int -> Prod transitions when different) ---")
    diff_df = df[post_diff_mask].copy()
    if len(diff_df) > 0:
        # Create transition column
        diff_df["transition"] = (
            diff_df["int_post_postprocessing_class"].astype(str) +
            " -> " +
            diff_df["prod_post_postprocessing_class"].astype(str)
        )
        transition_counts = diff_df["transition"].value_counts()
        print(transition_counts.to_string())
        summary["error_transitions"] = transition_counts.to_dict()
        print()

        # Confusion matrix style summary
        print("--- Confusion Matrix (Int vs Prod when different) ---")
        confusion = pd.crosstab(
            diff_df["int_post_postprocessing_class"],
            diff_df["prod_post_postprocessing_class"],
            margins=True,
            margins_name="Total"
        )
        print(confusion.to_string())
        summary["confusion_matrix"] = confusion.to_dict()
        print()

        # Statistics for rows where predictions differ
        print("--- Statistics for Rows with Different Predictions ---")
        stat_cols = ["int_sog_mean", "prod_sog_mean", "int_num_messages", "prod_num_messages",
                     "int_avg_time_gap_seconds", "prod_avg_time_gap_seconds"]
        available_stat_cols = [c for c in stat_cols if c in diff_df.columns]
        if available_stat_cols:
            diff_stats = diff_df[available_stat_cols].describe()
            print(diff_stats.to_string())
            summary["diff_row_statistics"] = diff_stats.to_dict()
        print()

        # Compare statistics between same vs different predictions
        print("--- Comparing Stats: Same Predictions vs Different Predictions ---")
        same_df = df[~post_diff_mask]
        comparison_stats = []
        for col in ["int_sog_mean", "int_num_messages", "int_avg_time_gap_seconds"]:
            if col in df.columns:
                same_mean = same_df[col].mean() if len(same_df) > 0 else np.nan
                diff_mean = diff_df[col].mean() if len(diff_df) > 0 else np.nan
                comparison_stats.append({
                    "metric": col,
                    "same_pred_mean": round(same_mean, 4) if pd.notna(same_mean) else None,
                    "diff_pred_mean": round(diff_mean, 4) if pd.notna(diff_mean) else None,
                    "difference": round(diff_mean - same_mean, 4) if pd.notna(same_mean) and pd.notna(diff_mean) else None,
                })
        if comparison_stats:
            comp_df = pd.DataFrame(comparison_stats)
            print(comp_df.to_string(index=False))
            summary["same_vs_diff_comparison"] = comparison_stats  # type: ignore
        print()

        # Per-transition distributional analysis
        print("=" * 80)
        print("PER-TRANSITION DISTRIBUTIONAL ANALYSIS (Int - Prod Differences)")
        print("=" * 80)
        print()

        # Define pairs to calculate differences: (int_col, prod_col, display_name)
        diff_pairs = [
            ("int_sog_mean", "prod_sog_mean", "SOG Mean Diff"),
            ("int_sog_min", "prod_sog_min", "SOG Min Diff"),
            ("int_sog_max", "prod_sog_max", "SOG Max Diff"),
            ("int_sog_std", "prod_sog_std", "SOG Std Diff"),
            ("int_cog_mean", "prod_cog_mean", "COG Mean Diff"),
            ("int_cog_std", "prod_cog_std", "COG Std Diff"),
            ("int_num_messages", "prod_num_messages", "Num Messages Diff"),
            ("int_avg_time_gap_seconds", "prod_avg_time_gap_seconds", "Avg Time Gap Diff (s)"),
        ]

        # Calculate difference columns
        for int_col, prod_col, _ in diff_pairs:
            if int_col in diff_df.columns and prod_col in diff_df.columns:
                diff_df[f"diff_{int_col.replace('int_', '')}"] = diff_df[int_col] - diff_df[prod_col]

        # Also include overlap percentage (not a diff)
        standalone_cols = [
            ("overlap_percentage_of_union", "Overlap % of Union"),
            ("num_overlapping_timestamps", "Num Overlapping Timestamps"),
        ]

        transition_stats_list = []
        for transition_type in transition_counts.index[:15]:  # Top 15 transitions
            trans_df = diff_df[diff_df["transition"] == transition_type]
            n_samples = len(trans_df)

            print(f"--- {transition_type} (n={n_samples}) ---")

            trans_stats = {"transition": transition_type, "count": n_samples}

            # Report on the differences (int - prod)
            print("  Int - Prod Differences:")
            for int_col, prod_col, display_name in diff_pairs:
                diff_col = f"diff_{int_col.replace('int_', '')}"
                if diff_col in trans_df.columns:
                    values = trans_df[diff_col].dropna()
                    if len(values) > 0:
                        mean_val = values.mean()
                        std_val = values.std()
                        median_val = values.median()

                        print(f"    {display_name}: mean={mean_val:+.2f}, std={std_val:.2f}, median={median_val:+.2f}")

                        trans_stats[f"{diff_col}_mean"] = round(mean_val, 4)
                        trans_stats[f"{diff_col}_std"] = round(std_val, 4) if pd.notna(std_val) else None
                        trans_stats[f"{diff_col}_median"] = round(median_val, 4)

            # Report standalone columns
            print("  Other Metrics:")
            for col, display_name in standalone_cols:
                if col in trans_df.columns:
                    values = trans_df[col].dropna()
                    if len(values) > 0:
                        mean_val = values.mean()
                        print(f"    {display_name}: mean={mean_val:.2f}")
                        trans_stats[f"{col}_mean"] = round(mean_val, 4)

            transition_stats_list.append(trans_stats)
            print()

        summary["per_transition_stats"] = transition_stats_list  # type: ignore

        # Print comparison summary table
        if transition_stats_list:
            print("=" * 80)
            print("TRANSITION COMPARISON SUMMARY (Int - Prod Differences)")
            print("=" * 80)
            print()

            # Build a compact comparison table showing differences
            comparison_rows = []
            for ts in transition_stats_list:
                row = {
                    "Transition": ts["transition"],
                    "N": ts["count"],
                    "SOG Diff": ts.get("diff_sog_mean_mean"),
                    "Msgs Diff": ts.get("diff_num_messages_mean"),
                    "Time Gap Diff": ts.get("diff_avg_time_gap_seconds_mean"),
                    "Overlap %": ts.get("overlap_percentage_of_union_mean"),
                }
                comparison_rows.append(row)

            comparison_df = pd.DataFrame(comparison_rows)
            # Format numeric columns
            for col in ["SOG Diff", "Msgs Diff", "Time Gap Diff", "Overlap %"]:
                if col in comparison_df.columns:
                    comparison_df[col] = comparison_df[col].apply(
                        lambda x: f"{x:+.2f}" if pd.notna(x) else "N/A"
                    )
            print(comparison_df.to_string(index=False))
            print()

        # Identify which metrics have large magnitude differences for each transition
        print("=" * 80)
        print("LARGE MAGNITUDE DIFFERENCES BY TRANSITION")
        print("(Metrics ranked by |mean(int - prod)|)")
        print("=" * 80)
        print()

        diff_col_names = [f"diff_{int_col.replace('int_', '')}" for int_col, _, _ in diff_pairs]

        notable_findings = []

        for transition_type in transition_counts.index[:15]:
            trans_df = diff_df[diff_df["transition"] == transition_type]
            n_samples = len(trans_df)

            if n_samples < 3:  # Skip very small groups
                continue

            metric_magnitudes = []

            for diff_col in diff_col_names:
                if diff_col not in trans_df.columns:
                    continue

                values = trans_df[diff_col].dropna()
                if len(values) == 0:
                    continue

                mean_diff = values.mean()
                abs_mean_diff = abs(mean_diff)
                median_diff = values.median()
                std_diff = values.std()

                metric_name = diff_col.replace("diff_", "").replace("_", " ").title()
                direction = "int > prod" if mean_diff > 0 else "prod > int"

                metric_magnitudes.append({
                    "metric": metric_name,
                    "mean_diff": mean_diff,
                    "abs_mean_diff": abs_mean_diff,
                    "median_diff": median_diff,
                    "std_diff": std_diff,
                    "direction": direction,
                })

                notable_findings.append({
                    "transition": transition_type,
                    "count": n_samples,
                    "metric": metric_name,
                    "mean_diff": round(mean_diff, 2),
                    "abs_mean_diff": round(abs_mean_diff, 2),
                    "median_diff": round(median_diff, 2),
                    "std_diff": round(std_diff, 2) if pd.notna(std_diff) else None,
                    "direction": direction,
                })

            if metric_magnitudes:
                print(f"--- {transition_type} (n={n_samples}) ---")
                # Sort by absolute magnitude, show top metrics
                for mm in sorted(metric_magnitudes, key=lambda x: x["abs_mean_diff"], reverse=True):
                    print(f"  {mm['metric']}: mean={mm['mean_diff']:+.2f}, "
                          f"median={mm['median_diff']:+.2f}, std={mm['std_diff']:.2f} ({mm['direction']})")
                print()

        if notable_findings:
            summary["large_magnitude_differences"] = notable_findings

            # Save findings to CSV, sorted by absolute magnitude within each transition
            if output_dir:
                findings_df = pd.DataFrame(notable_findings)
                findings_df = findings_df.sort_values(
                    ["transition", "abs_mean_diff"],
                    ascending=[True, False]
                )
                notable_csv = output_dir / "large_magnitude_differences.csv"
                findings_df.to_csv(notable_csv, index=False)
                print(f"Saved large magnitude differences to {notable_csv}")

        # Save per-transition stats to CSV
        if output_dir and transition_stats_list:
            trans_stats_df = pd.DataFrame(transition_stats_list)
            trans_stats_csv = output_dir / "per_transition_statistics.csv"
            trans_stats_df.to_csv(trans_stats_csv, index=False)
            print(f"Saved per-transition statistics to {trans_stats_csv}")

    else:
        print("No differences found in post-processing predictions.")
        summary["error_transitions"] = {}

    # Save summary if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save text summary
        summary_file = output_dir / "classification_summary.txt"
        with open(summary_file, "w") as f:
            f.write("CLASSIFICATION DIFFERENCE SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total rows: {total_rows}\n")
            f.write(f"Post-processing differences: {post_diff_count} ({100*post_diff_count/total_rows:.1f}%)\n")
            f.write(f"Pre-processing differences: {pre_diff_count} ({100*pre_diff_count/total_rows:.1f}%)\n\n")

            f.write("Int Post-Processing Distribution:\n")
            f.write(int_post_dist.to_string() + "\n\n")

            f.write("Prod Post-Processing Distribution:\n")
            f.write(prod_post_dist.to_string() + "\n\n")

            if len(diff_df) > 0:
                f.write("Error Transitions (Int -> Prod):\n")
                f.write(transition_counts.to_string() + "\n\n")

                f.write("Confusion Matrix:\n")
                f.write(confusion.to_string() + "\n")

        print(f"Saved classification summary to {summary_file}")

        # Save rows with differences to CSV
        if len(diff_df) > 0:
            diff_csv = output_dir / "classification_differences.csv"
            diff_df.to_csv(diff_csv, index=False)
            print(f"Saved {len(diff_df)} differing rows to {diff_csv}")

        # Save transition counts
        if len(diff_df) > 0:
            transitions_csv = output_dir / "error_transitions.csv"
            transition_counts.to_frame("count").reset_index().rename(
                columns={"index": "transition"}
            ).to_csv(transitions_csv, index=False)
            print(f"Saved error transitions to {transitions_csv}")

    return summary


def analyze_differences(csv_path: str, threshold_percentile: float = 90.0) -> None:
    """Analyze differences between int and prod statistics."""
    df = pd.read_csv(csv_path)

    # Define pairs of int/prod columns to compare
    comparison_pairs = [
        ("sog_mean", "int_sog_mean", "prod_sog_mean"),
        ("sog_min", "int_sog_min", "prod_sog_min"),
        ("sog_max", "int_sog_max", "prod_sog_max"),
        ("sog_std", "int_sog_std", "prod_sog_std"),
        ("cog_mean", "int_cog_mean", "prod_cog_mean"),
        ("cog_min", "int_cog_min", "prod_cog_min"),
        ("cog_max", "int_cog_max", "prod_cog_max"),
        ("cog_std", "int_cog_std", "prod_cog_std"),
        ("num_messages", "int_num_messages", "prod_num_messages"),
        ("avg_time_gap_seconds", "int_avg_time_gap_seconds", "prod_avg_time_gap_seconds"),
    ]

    print("=" * 80)
    print("DIFFERENCE ANALYSIS: Int vs Prod Statistics")
    print("=" * 80)
    print()

    big_differences = []

    for field_name, int_col, prod_col in comparison_pairs:
        if int_col not in df.columns or prod_col not in df.columns:
            continue

        int_values = pd.to_numeric(df[int_col], errors='coerce')
        prod_values = pd.to_numeric(df[prod_col], errors='coerce')

        # Calculate absolute differences
        abs_diff = np.abs(int_values - prod_values)

        # Calculate relative differences (as percentage)
        # Use mean of int and prod to avoid division by zero
        mean_values = (int_values + prod_values) / 2
        rel_diff = np.where(
            mean_values != 0,
            (abs_diff / np.abs(mean_values)) * 100,
            np.where(abs_diff != 0, np.inf, 0)
        )

        # Calculate statistics
        valid_mask = ~(int_values.isna() | prod_values.isna())
        if valid_mask.sum() == 0:
            continue

        abs_diff_valid = abs_diff[valid_mask]
        rel_diff_valid = rel_diff[valid_mask]

        # Find threshold for "big differences"
        abs_threshold = np.percentile(abs_diff_valid, threshold_percentile)
        rel_threshold = np.percentile(rel_diff_valid[rel_diff_valid != np.inf], threshold_percentile)

        # Find rows with big differences
        big_diff_mask = (abs_diff > abs_threshold) | (
            (rel_diff > rel_threshold) & (rel_diff != np.inf)
        )
        big_diff_rows = df[big_diff_mask]

        print(f"Field: {field_name}")
        print(f"  Mean absolute difference: {abs_diff_valid.mean():.4f}")
        print(f"  Median absolute difference: {abs_diff_valid.median():.4f}")
        print(f"  Max absolute difference: {abs_diff_valid.max():.4f}")
        print(f"  Mean relative difference (%): {rel_diff_valid[rel_diff_valid != np.inf].mean():.2f}%")
        print(f"  Median relative difference (%): {np.median(rel_diff_valid[rel_diff_valid != np.inf]):.2f}%")
        print(f"  Max relative difference (%): {rel_diff_valid[rel_diff_valid != np.inf].max():.2f}%")
        print(f"  Threshold ({threshold_percentile}th percentile): abs={abs_threshold:.4f}, rel={rel_threshold:.2f}%")
        print(f"  Number of big differences: {big_diff_mask.sum()}")

        if big_diff_mask.sum() > 0:
            print("  Subpath pairs with big differences:")
            for idx, row in big_diff_rows.iterrows():
                int_val = row[int_col]
                prod_val = row[prod_col]
                abs_d = abs_diff.iloc[idx]
                rel_d = rel_diff[idx] if rel_diff[idx] != np.inf else "inf"
                int_subpath_id = row.get("int_subpath_id", "N/A")
                prod_subpath_id = row.get("prod_subpath_id", "N/A")

                # Determine which is higher
                if pd.notna(int_val) and pd.notna(prod_val):
                    if int_val > prod_val:
                        higher = "int"
                    elif prod_val > int_val:
                        higher = "prod"
                    else:
                        higher = "equal"
                else:
                    higher = "unknown"

                print(f"    - Int: {int_subpath_id}, Prod: {prod_subpath_id}")
                print(f"      Int={int_val:.4f}, Prod={prod_val:.4f}, AbsDiff={abs_d:.4f}, RelDiff={rel_d:.2f}%, Higher={higher}")
                big_differences.append({
                    "field": field_name,
                    "int_subpath_id": int_subpath_id,
                    "prod_subpath_id": prod_subpath_id,
                    "int_value": float(int_val) if pd.notna(int_val) else None,
                    "prod_value": float(prod_val) if pd.notna(prod_val) else None,
                    "abs_diff": float(abs_d) if pd.notna(abs_d) else None,
                    "rel_diff": float(rel_d) if rel_d != np.inf and pd.notna(rel_d) else None,
                    "higher": higher,
                })
        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    if big_differences:
        print(f"Total big differences found: {len(big_differences)}")

        # Count which is higher more often
        int_higher = sum(1 for d in big_differences if d.get("higher") == "int")
        prod_higher = sum(1 for d in big_differences if d.get("higher") == "prod")
        print(f"Int higher: {int_higher}, Prod higher: {prod_higher}")

        print("\nAll big differences:")
        for diff in big_differences:
            abs_d = diff['abs_diff'] if diff['abs_diff'] is not None else 0
            rel_d = diff['rel_diff'] if diff['rel_diff'] is not None else 0
            higher = diff.get('higher', 'unknown')
            print(f"  {diff['field']}: Int={diff['int_subpath_id']}, Prod={diff['prod_subpath_id']}, "
                  f"AbsDiff={abs_d:.4f}, RelDiff={rel_d:.2f}%, Higher={higher}")
    else:
        print("No big differences found.")

    return big_differences


def main():
    parser = argparse.ArgumentParser(description="Analyze differences between int and prod statistics")
    parser.add_argument("csv", help="Path to comparison_statistics.csv")
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=90.0,
        help="Percentile threshold for identifying big differences (default: 90.0)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Directory to save summary files. If not provided, uses the same directory as the input CSV.",
    )
    parser.add_argument(
        "--skip-stats",
        action="store_true",
        help="Skip the statistical difference analysis",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate distribution plots comparing int vs prod for each transition type",
    )
    parser.add_argument(
        "--top-n-transitions",
        type=int,
        default=10,
        help="Number of top transitions to plot (default: 10)",
    )
    args = parser.parse_args()

    # Determine output directory
    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data once
    df = pd.read_csv(args.csv)

    # Run classification difference analysis
    analyze_classification_differences(df, output_dir)

    # Generate distribution plots if requested
    if args.plot:
        print()
        print("=" * 80)
        print("GENERATING DISTRIBUTION PLOTS")
        print("=" * 80)
        plot_transition_distributions(df, output_dir, args.top_n_transitions)

    # Run statistical difference analysis
    if not args.skip_stats:
        print()
        analyze_differences(args.csv, args.threshold_percentile)


if __name__ == "__main__":
    main()
