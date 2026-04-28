"""
Visualization functions for lead-lag analysis results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List
from pathlib import Path


class LeadLagVisualizer:
    """Visualize lead-lag analysis results."""

    def __init__(self, output_dir: str = "results"):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)

    def plot_delay_matrix(
        self,
        delays_df: pd.DataFrame,
        coherence_df: pd.DataFrame,
        title: str = "Lead-Lag Delays",
        save_path: Optional[str] = None,
        min_coherence: float = 0.0
    ) -> plt.Figure:
        """
        Plot heatmap of delay matrix with coherence overlay.

        Args:
            delays_df: Delay matrix (seconds)
            coherence_df: Coherence matrix (0-1)
            title: Plot title
            save_path: Optional path to save figure
            min_coherence: Minimum coherence to display (mask low coherence values)

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Mask low coherence values
        masked_delays = delays_df.copy()
        mask = coherence_df < min_coherence
        masked_delays[mask] = np.nan

        # Plot delays
        sns.heatmap(
            masked_delays,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            cbar_kws={'label': 'Delay (seconds)'},
            ax=ax1,
            vmin=-masked_delays.abs().max().max(),
            vmax=masked_delays.abs().max().max()
        )
        ax1.set_title(f'{title}\n(Positive = Row leads Column)')
        ax1.set_xlabel('Asset')
        ax1.set_ylabel('Asset')

        # Plot coherence
        sns.heatmap(
            coherence_df,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Coherence'},
            ax=ax2
        )
        ax2.set_title(f'{title} - Coherence\n(Higher = Stronger Relationship)')
        ax2.set_xlabel('Asset')
        ax2.set_ylabel('Asset')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")

        return fig

    def plot_leadership_scores(
        self,
        scores: pd.Series,
        title: str = "Asset Leadership Scores",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot leadership scores as a bar chart.

        Args:
            scores: Series with leadership scores
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Sort scores
        scores_sorted = scores.sort_values(ascending=False)

        # Create color palette (positive = green, negative = red)
        colors = ['green' if x > 0 else 'red' for x in scores_sorted.values]

        # Plot
        bars = ax.barh(range(len(scores_sorted)), scores_sorted.values, color=colors, alpha=0.7)
        ax.set_yticks(range(len(scores_sorted)))
        ax.set_yticklabels(scores_sorted.index)
        ax.set_xlabel('Leadership Score (seconds)')
        ax.set_title(title + '\n(Positive = Leads, Negative = Lags)')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")

        return fig

    def plot_relationship_network(
        self,
        relationships_df: pd.DataFrame,
        title: str = "Lead-Lag Network",
        save_path: Optional[str] = None,
        top_n: int = 10
    ) -> plt.Figure:
        """
        Plot lead-lag relationships as a network diagram.

        Args:
            relationships_df: DataFrame with columns: leader, lagger, delay_sec, coherence
            title: Plot title
            save_path: Optional path to save figure
            top_n: Number of top relationships to display

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # Take top N relationships
        top_relationships = relationships_df.head(top_n)

        if top_relationships.empty:
            ax.text(0.5, 0.5, 'No significant relationships found',
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(title)
            return fig

        # Get unique assets
        assets = list(set(top_relationships['leader'].unique()) |
                      set(top_relationships['lagger'].unique()))
        n_assets = len(assets)

        # Create circular layout
        angles = np.linspace(0, 2 * np.pi, n_assets, endpoint=False)
        positions = {asset: (np.cos(angle), np.sin(angle))
                     for asset, angle in zip(assets, angles)}

        # Plot nodes
        for asset, (x, y) in positions.items():
            ax.scatter(x, y, s=1000, c='lightblue', edgecolors='black', zorder=3)
            ax.text(x, y, asset, ha='center', va='center', fontsize=12, fontweight='bold')

        # Plot edges
        for _, row in top_relationships.iterrows():
            leader_pos = positions[row['leader']]
            lagger_pos = positions[row['lagger']]

            # Arrow from leader to lagger
            dx = lagger_pos[0] - leader_pos[0]
            dy = lagger_pos[1] - leader_pos[1]

            # Color and width based on coherence
            coherence = row['coherence']
            width = 1 + 4 * coherence  # 1-5 width range
            alpha = 0.3 + 0.7 * coherence

            ax.arrow(
                leader_pos[0], leader_pos[1],
                dx * 0.85, dy * 0.85,  # Shorten arrow to not overlap nodes
                head_width=0.1,
                head_length=0.05,
                fc=f'C0',
                ec=f'C0',
                alpha=alpha,
                linewidth=width,
                length_includes_head=True,
                zorder=2
            )

            # Add delay label at midpoint
            mid_x = (leader_pos[0] + lagger_pos[0]) / 2
            mid_y = (leader_pos[1] + lagger_pos[1]) / 2
            ax.text(mid_x, mid_y, f"{row['delay_sec']:.1f}s",
                    fontsize=8, ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title + f'\n(Top {top_n} relationships, arrow thickness = coherence)')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")

        return fig

    def plot_time_series_with_lag(
        self,
        data_dict: dict,
        asset1: str,
        asset2: str,
        delay_seconds: float,
        window: Optional[Tuple[int, int]] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot two time series overlaid, with one shifted by the detected lag.

        Args:
            data_dict: Dictionary mapping asset to OHLCV DataFrame
            asset1: Leader asset
            asset2: Lagger asset
            delay_seconds: Delay in seconds
            window: Optional (start_idx, end_idx) to zoom in
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # Get data
        df1 = data_dict[asset1]
        df2 = data_dict[asset2]

        if window:
            start, end = window
            df1 = df1.iloc[start:end]
            df2 = df2.iloc[start:end]

        # Plot original
        ax1 = axes[0]
        ax1.plot(df1.index, df1['close'], label=asset1, color='blue', alpha=0.7)
        ax1_twin = ax1.twinx()
        ax1_twin.plot(df2.index, df2['close'], label=asset2, color='red', alpha=0.7)

        ax1.set_ylabel(f'{asset1} Price', color='blue')
        ax1_twin.set_ylabel(f'{asset2} Price', color='red')
        ax1.set_title('Original Time Series')
        ax1.grid(alpha=0.3)

        # Plot with lag-adjusted
        ax2 = axes[1]

        # Shift asset2 by delay
        time_shift = pd.Timedelta(seconds=delay_seconds)
        df2_shifted = df2.copy()
        df2_shifted.index = df2_shifted.index - time_shift

        # Re-align to common index
        common_index = df1.index.intersection(df2_shifted.index)

        ax2.plot(common_index, df1.loc[common_index, 'close'],
                 label=asset1, color='blue', alpha=0.7)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(common_index, df2_shifted.loc[common_index, 'close'],
                      label=f'{asset2} (shifted {delay_seconds:.1f}s)', color='red', alpha=0.7)

        ax2.set_ylabel(f'{asset1} Price', color='blue')
        ax2_twin.set_ylabel(f'{asset2} Price (shifted)', color='red')
        ax2.set_title(f'Lag-Adjusted ({asset1} leads {asset2} by {delay_seconds:.1f}s)')
        ax2.set_xlabel('Time')
        ax2.grid(alpha=0.3)

        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")

        return fig

    def create_summary_report(
        self,
        results: dict,
        output_basename: str = "lead_lag_analysis",
        band_info: Optional[dict] = None
    ):
        """
        Create a comprehensive summary report with all visualizations.

        Args:
            results: Dictionary mapping band name to (delays_df, coherence_df, relationships_df, scores)
            output_basename: Base name for output files
            band_info: Optional dictionary mapping band name to FrequencyBand object for period information
        """
        print(f"\nGenerating visualizations in {self.output_dir}/")

        for band_name, (delays_df, coherence_df, relationships_df, scores) in results.items():
            # Clean band name for filename
            clean_name = band_name.replace(' ', '_').lower()

            # Get period information if available
            period_info = ""
            if band_info and band_name in band_info:
                band = band_info[band_name]
                p_min, p_max = band.period_range
                # Format period range nicely
                if p_max < 120:  # Less than 2 minutes
                    period_info = f" ({p_min:.0f}s - {p_max:.0f}s)"
                elif p_max < 7200:  # Less than 2 hours
                    period_info = f" ({p_min/60:.1f}min - {p_max/60:.1f}min)"
                else:
                    period_info = f" ({p_min/3600:.1f}hr - {p_max/3600:.1f}hr)"

            # Delay matrix
            self.plot_delay_matrix(
                delays_df,
                coherence_df,
                title=f"Lead-Lag Analysis - {band_name}{period_info}",
                save_path=self.output_dir / f"{output_basename}_{clean_name}_delays.png",
                min_coherence=0.2
            )
            plt.close()

            # Leadership scores
            if not scores.empty:
                self.plot_leadership_scores(
                    scores,
                    title=f"Leadership Scores - {band_name}{period_info}",
                    save_path=self.output_dir / f"{output_basename}_{clean_name}_leadership.png"
                )
                plt.close()

            # Network diagram
            if not relationships_df.empty:
                self.plot_relationship_network(
                    relationships_df,
                    title=f"Lead-Lag Network - {band_name}{period_info}",
                    save_path=self.output_dir / f"{output_basename}_{clean_name}_network.png",
                    top_n=10
                )
                plt.close()

        print(f"All visualizations saved to {self.output_dir}/")


if __name__ == '__main__':
    # Example usage
    print("Visualizer module - import and use with analyzer results")
