"""
visualizer.py
=============
All visualisations for the football impact rating system.

The aesthetic is deliberately chosen: dark background with green accents
mirrors the StatsBomb / Opta / FBref visual language that football analytics
audiences recognise as authoritative. It also improves readability of scatter
plots with many overlapping points.

Each chart is designed around a football communication purpose, not a data
science purpose:
- The radar chart is the universal scout report format — every scout from
  Europe to South America understands a radar chart
- The position scatter reveals which quadrant a player occupies (the "Complete
  Defender" quadrant, the "Regista" zone, etc.)
- The similarity network makes a scout's intuitive "players in the same mould"
  claim mathematically concrete and auditable

All charts save to outputs/ and return the figure object for Jupyter embedding.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script execution
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch

logger = logging.getLogger(__name__)

# ── Aesthetic constants ────────────────────────────────────────────────────────
GREEN = "#00FF87"       # primary StatsBomb-style accent
AMBER = "#FFB800"       # secondary highlight
RED = "#FF4444"         # weakness / negative
WHITE = "#FFFFFF"
GREY = "#888888"
DARK_BG = "#0D1117"     # github dark — slightly less harsh than pure black
CLUSTER_PALETTE = ["#00FF87", "#FFB800", "#FF6B9D", "#00C8FF", "#FF8C00", "#C0FF00"]

OUTPUT_DIR = Path("outputs")


def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _save_and_return(fig: plt.Figure, filename: str) -> plt.Figure:
    _ensure_output_dir()
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    logger.info("Saved chart: %s", path)
    return fig


class FootballVisualizer:
    """
    Generates football analytics visualisations with a StatsBomb-inspired aesthetic.

    All methods:
    - Accept a position-specific or full DataFrame
    - Save to outputs/ directory as PNG
    - Return the matplotlib Figure object (for notebook embedding)
    - Use dark background with green accent as the primary visual language
    """

    # ── CHART 1: Radar / Spider Chart ─────────────────────────────────────────
    def radar_chart(
        self,
        player_name: str,
        df: pd.DataFrame,
        filename: Optional[str] = None,
    ) -> plt.Figure:
        """
        Pentagon radar showing all component scores vs. position average.

        The radar chart is the standard format for scout reports globally.
        Two layers: player scores (green fill) + position average (white outline).
        This immediately shows a player's profile shape — are they a balanced
        contributor or do they spike in one dimension?

        Parameters
        ----------
        player_name : str
            Name to look up in df.
        df : pd.DataFrame
            Scored DataFrame with normalised component columns.
        filename : str, optional
            Output filename. Auto-generated if None.
        """
        mask = df["player_name"].str.lower().str.contains(player_name.lower(), na=False)
        if not mask.any():
            logger.warning("Player '%s' not found for radar chart", player_name)
            return plt.figure()

        player = df[mask].iloc[0]
        position = player.get("position", "")

        # Select normalised component columns
        norm_cols = [c for c in df.columns if c.endswith("_normalized")
                     and "GK" not in c or (position == "GK" and "GK" in c)]
        if not norm_cols:
            norm_cols = [c for c in df.columns if c.endswith("_normalized")]

        # Clean labels
        labels = [c.replace("_normalized", "").replace("_", "\n") for c in norm_cols]
        n = len(labels)
        if n < 3:
            logger.warning("Too few components for radar chart")
            return plt.figure()

        player_vals = np.array([float(player.get(c, 0)) for c in norm_cols])
        # Position average
        pos_df = df[df["position"] == position] if "position" in df.columns else df
        pos_avg = np.array([float(pos_df[c].mean()) for c in norm_cols])

        # Angles for radar
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        angles += angles[:1]  # close polygon
        player_vals = np.append(player_vals, player_vals[0])
        pos_avg = np.append(pos_avg, pos_avg[0])

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True},
                               facecolor=DARK_BG)
        ax.set_facecolor(DARK_BG)

        # Grid
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["25", "50", "75", "100"], color=GREY, fontsize=8)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, color=WHITE, fontsize=11, fontweight="bold")
        ax.tick_params(axis="y", colors=GREY)
        ax.grid(color=GREY, linestyle="--", linewidth=0.5, alpha=0.4)
        ax.spines["polar"].set_color(GREY)
        ax.spines["polar"].set_alpha(0.3)

        # Position average (white outline)
        ax.plot(angles, pos_avg, color=WHITE, linewidth=1.5, linestyle="--",
                alpha=0.6, label=f"{position} Average")
        ax.fill(angles, pos_avg, color=WHITE, alpha=0.05)

        # Player (green fill)
        ax.plot(angles, player_vals, color=GREEN, linewidth=2.5, label=player_name)
        ax.fill(angles, player_vals, color=GREEN, alpha=0.25)

        # Scatter dots on vertices
        ax.scatter(angles[:-1], player_vals[:-1], color=GREEN, s=60, zorder=5)

        # Score text in centre
        score = player.get("impact_score", 0)
        club = player.get("club", "")
        ax.text(0, 0, f"{score:.0f}", ha="center", va="center",
                fontsize=28, fontweight="bold", color=GREEN,
                transform=ax.transData)

        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
                  facecolor=DARK_BG, edgecolor=GREY, labelcolor=WHITE, fontsize=10)

        title = f"{player['player_name']}  |  {club}  |  {position}\nImpact Score: {score:.1f}/100"
        fig.suptitle(title, fontsize=14, color=WHITE, fontweight="bold", y=0.95)

        fname = filename or f"radar_{player_name.lower().replace(' ', '_')}.png"
        return _save_and_return(fig, fname)

    # ── CHART 2: Position Scatter ──────────────────────────────────────────────
    def position_scatter(
        self,
        df: pd.DataFrame,
        position: str,
        x_component: str,
        y_component: str,
        filename: Optional[str] = None,
    ) -> plt.Figure:
        """
        Scatter plot of two components for all players in a position.

        Quadrant lines at position median divide the space into four zones
        labelled with football meaning (e.g., high DAQ + high PPI = Complete Defender).
        Players in the top 10% impact score are labelled by name — these are the
        analysts' 'interesting cases'.

        Parameters
        ----------
        df : pd.DataFrame
            Position-specific DataFrame with component scores and archetype labels.
        position : str
            Position label for axis context.
        x_component : str
            Column name for x-axis (e.g., 'DAQ').
        y_component : str
            Column name for y-axis (e.g., 'PPI').
        """
        pos_df = df[df["position"] == position].copy() if "position" in df.columns else df.copy()
        if pos_df.empty:
            logger.warning("No players for position %s in scatter", position)
            return plt.figure()

        fig, ax = plt.subplots(figsize=(12, 9), facecolor=DARK_BG)
        ax.set_facecolor(DARK_BG)

        x = pos_df[x_component].fillna(0)
        y = pos_df[y_component].fillna(0)
        x_med = x.median()
        y_med = y.median()

        # Colour by archetype
        archetypes = pos_df["archetype_label"].unique() if "archetype_label" in pos_df.columns else ["All"]
        archetype_colors = {arch: CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
                            for i, arch in enumerate(archetypes)}

        for arch in archetypes:
            if "archetype_label" in pos_df.columns:
                subset = pos_df[pos_df["archetype_label"] == arch]
            else:
                subset = pos_df
            ax.scatter(x[subset.index], y[subset.index],
                       color=archetype_colors.get(arch, GREEN),
                       alpha=0.7, s=60, label=arch, zorder=3)

        # Quadrant lines
        ax.axvline(x_med, color=GREY, linestyle="--", linewidth=1, alpha=0.6)
        ax.axhline(y_med, color=GREY, linestyle="--", linewidth=1, alpha=0.6)

        # Quadrant labels
        x_rng = x.max() - x.min()
        y_rng = y.max() - y.min()
        pad_x = x_rng * 0.03
        pad_y = y_rng * 0.03

        quad_labels = {
            (x.max() - pad_x, y.max() - pad_y, "right", "top"): f"High {x_component}\nHigh {y_component}",
            (x.min() + pad_x, y.max() - pad_y, "left", "top"): f"Low {x_component}\nHigh {y_component}",
            (x.max() - pad_x, y.min() + pad_y, "right", "bottom"): f"High {x_component}\nLow {y_component}",
            (x.min() + pad_x, y.min() + pad_y, "left", "bottom"): f"Low {x_component}\nLow {y_component}",
        }
        for (qx, qy, ha, va), label in quad_labels.items():
            ax.text(qx, qy, label, color=GREY, fontsize=8, ha=ha, va=va, alpha=0.7,
                    style="italic")

        # Label top 10% impact scorers
        if "impact_score" in pos_df.columns:
            threshold = pos_df["impact_score"].quantile(0.90)
            top_players = pos_df[pos_df["impact_score"] >= threshold]
            for _, row in top_players.iterrows():
                ax.annotate(
                    row["player_name"],
                    (x[row.name], y[row.name]),
                    xytext=(6, 4),
                    textcoords="offset points",
                    color=WHITE,
                    fontsize=7,
                    fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=2, foreground=DARK_BG)],
                )

        ax.set_xlabel(x_component, color=WHITE, fontsize=12, fontweight="bold")
        ax.set_ylabel(y_component, color=WHITE, fontsize=12, fontweight="bold")
        ax.tick_params(colors=GREY)
        ax.spines["bottom"].set_color(GREY)
        ax.spines["left"].set_color(GREY)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        legend = ax.legend(facecolor=DARK_BG, edgecolor=GREY, labelcolor=WHITE,
                           fontsize=9, loc="best")

        fig.suptitle(f"{position}s: {x_component} vs {y_component}  |  Top 10% labelled",
                     color=WHITE, fontsize=14, fontweight="bold")

        fname = filename or f"{position.lower()}_scatter_{x_component.lower()}_{y_component.lower()}.png"
        return _save_and_return(fig, fname)

    # ── CHART 3: Impact Score Distribution ────────────────────────────────────
    def impact_score_distribution(
        self,
        df: pd.DataFrame,
        position: str,
        top_n: int = 20,
        filename: Optional[str] = None,
    ) -> plt.Figure:
        """
        Horizontal bar chart of top N players by impact score.

        Bars are coloured by archetype and annotated with the player's
        strongest component — giving scouts an instant read on WHY a player
        ranks where they do.

        Parameters
        ----------
        df : pd.DataFrame
        position : str
        top_n : int
            Number of players to display. Default 20.
        """
        pos_df = df[df["position"] == position].copy() if "position" in df.columns else df.copy()
        if "impact_score" not in pos_df.columns or pos_df.empty:
            logger.warning("No impact scores for %s", position)
            return plt.figure()

        top = pos_df.nlargest(top_n, "impact_score").copy()

        archetypes = top["archetype_label"].unique() if "archetype_label" in top.columns else ["All"]
        archetype_colors = {arch: CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
                            for i, arch in enumerate(archetypes)}

        fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.45)), facecolor=DARK_BG)
        ax.set_facecolor(DARK_BG)

        norm_cols = [c for c in top.columns if c.endswith("_normalized")]

        for i, (_, row) in enumerate(top.iterrows()):
            arch = row.get("archetype_label", "All")
            color = archetype_colors.get(arch, GREEN)
            score = row["impact_score"]

            ax.barh(i, score, color=color, alpha=0.85, height=0.7, zorder=3)

            # Strongest component label
            if norm_cols:
                norm_vals = {c: float(row.get(c, 0)) for c in norm_cols if pd.notna(row.get(c, np.nan))}
                if norm_vals:
                    strongest = max(norm_vals, key=norm_vals.get).replace("_normalized", "")
                    ax.text(score + 0.5, i, f"↑{strongest}", color=color, fontsize=7.5,
                            va="center", fontweight="bold")

            # Player name
            ax.text(0.5, i, f"  {row['player_name']}",
                    color=WHITE, fontsize=9, va="center", fontweight="bold")

        ax.set_yticks(range(len(top)))
        ax.set_yticklabels([""] * len(top))
        ax.invert_yaxis()
        ax.set_xlim(0, 110)
        ax.set_xlabel("Impact Score (0-100)", color=WHITE, fontsize=11)
        ax.tick_params(axis="x", colors=GREY)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(GREY)

        ax.axvline(pos_df["impact_score"].mean(), color=AMBER, linestyle="--",
                   linewidth=1.5, alpha=0.7, label="Position average")

        # Legend for archetypes
        patches = [mpatches.Patch(color=archetype_colors[a], label=a) for a in archetypes]
        ax.legend(handles=patches, facecolor=DARK_BG, edgecolor=GREY,
                  labelcolor=WHITE, fontsize=8, loc="lower right")

        fig.suptitle(f"Top {top_n} {position}s by Impact Score",
                     color=WHITE, fontsize=14, fontweight="bold")
        plt.tight_layout()

        fname = filename or f"{position.lower()}_impact_distribution.png"
        return _save_and_return(fig, fname)

    # ── CHART 4: Archetype Profile Heatmap ────────────────────────────────────
    def archetype_profile_heatmap(
        self,
        df: pd.DataFrame,
        position: str,
        filename: Optional[str] = None,
    ) -> plt.Figure:
        """
        Heatmap: archetypes as rows, components as columns, mean normalised
        scores as values.

        This is the 'fingerprint' view — it makes visible exactly what statistical
        signature distinguishes a Deep-Lying Playmaker from a Box-to-Box Warrior.
        Each cell is annotated with its value for precise reading.

        Parameters
        ----------
        df : pd.DataFrame
        position : str
        """
        pos_df = df[df["position"] == position].copy() if "position" in df.columns else df.copy()
        if "archetype_label" not in pos_df.columns:
            logger.warning("No archetype labels for heatmap")
            return plt.figure()

        norm_cols = [c for c in pos_df.columns if c.endswith("_normalized")]
        if not norm_cols:
            logger.warning("No normalized columns for heatmap")
            return plt.figure()

        clean_labels = {c: c.replace("_normalized", "").replace("_", " ") for c in norm_cols}
        heatmap_data = pos_df.groupby("archetype_label")[norm_cols].mean() * 100
        heatmap_data = heatmap_data.rename(columns=clean_labels)

        fig, ax = plt.subplots(figsize=(max(10, len(norm_cols) * 2), max(5, len(heatmap_data) * 1.2)),
                               facecolor=DARK_BG)
        ax.set_facecolor(DARK_BG)

        data = heatmap_data.values
        im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

        # Annotate cells
        for i in range(len(heatmap_data)):
            for j in range(len(norm_cols)):
                val = data[i, j]
                txt_color = "black" if 30 < val < 70 else WHITE
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        color=txt_color, fontsize=11, fontweight="bold")

        ax.set_xticks(range(len(norm_cols)))
        ax.set_xticklabels(list(heatmap_data.columns), color=WHITE, fontsize=10,
                           rotation=30, ha="right")
        ax.set_yticks(range(len(heatmap_data)))
        ax.set_yticklabels(heatmap_data.index, color=WHITE, fontsize=10)
        ax.tick_params(colors=GREY)

        cbar = plt.colorbar(im, ax=ax, shrink=0.7)
        cbar.ax.yaxis.set_tick_params(color=WHITE)
        cbar.set_label("Mean Score (0-100)", color=WHITE, fontsize=10)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=WHITE)

        fig.suptitle(f"{position} Archetype Fingerprints — Component Means",
                     color=WHITE, fontsize=14, fontweight="bold")
        plt.tight_layout()

        fname = filename or f"{position.lower()}_archetype_heatmap.png"
        return _save_and_return(fig, fname)

    # ── CHART 5: Multi-Player Comparison Radar ────────────────────────────────
    def player_comparison_spider(
        self,
        player_list: List[str],
        df: pd.DataFrame,
        filename: Optional[str] = None,
    ) -> plt.Figure:
        """
        Multi-player radar for up to 4 players simultaneously.

        Each player a different semi-transparent colour. Legend includes
        name, club, and impact score for immediate comparison context.

        Parameters
        ----------
        player_list : list of str
            Up to 4 player names.
        df : pd.DataFrame
            Scored DataFrame.
        """
        colors = [GREEN, AMBER, "#FF6B9D", "#00C8FF"]
        player_list = player_list[:4]

        norm_cols = [c for c in df.columns if c.endswith("_normalized")]
        if not norm_cols:
            logger.warning("No normalized columns for comparison spider")
            return plt.figure()

        labels = [c.replace("_normalized", "").replace("_", "\n") for c in norm_cols]
        n = len(labels)

        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True},
                               facecolor=DARK_BG)
        ax.set_facecolor(DARK_BG)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["25", "50", "75", "100"], color=GREY, fontsize=8)
        ax.grid(color=GREY, linestyle="--", linewidth=0.5, alpha=0.4)
        ax.spines["polar"].set_color(GREY)
        ax.spines["polar"].set_alpha(0.3)

        legend_handles = []
        for i, name in enumerate(player_list):
            mask = df["player_name"].str.lower().str.contains(name.lower(), na=False)
            if not mask.any():
                continue
            player = df[mask].iloc[0]
            # Use only norm_cols that this player actually has (non-NaN)
            player_norm_cols = [c for c in norm_cols if pd.notna(player.get(c, np.nan))]
            if not player_norm_cols:
                continue
            player_labels = [c.replace("_normalized", "").replace("_", "\n") for c in player_norm_cols]
            n_p = len(player_norm_cols)
            angles = np.linspace(0, 2 * np.pi, n_p, endpoint=False).tolist()
            angles += angles[:1]
            vals = np.array([float(player.get(c, 0)) for c in player_norm_cols])
            vals_closed = np.append(vals, vals[0])

            if i == 0:
                # Set axes labels based on first player's cols
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(player_labels, color=WHITE, fontsize=10, fontweight="bold")
            color = colors[i]

            ax.plot(angles, vals_closed, color=color, linewidth=2.5)
            ax.fill(angles, vals_closed, color=color, alpha=0.15)
            ax.scatter(angles[:-1], vals, color=color, s=55, zorder=5)

            score = player.get("impact_score", 0)
            club = player.get("club", "")
            legend_handles.append(
                mpatches.Patch(color=color,
                               label=f"{player['player_name']}  |  {club}  |  {score:.0f}/100")
            )

        ax.legend(handles=legend_handles, loc="upper right",
                  bbox_to_anchor=(1.35, 1.1), facecolor=DARK_BG,
                  edgecolor=GREY, labelcolor=WHITE, fontsize=9)

        fig.suptitle("Player Comparison Radar", color=WHITE, fontsize=14,
                     fontweight="bold", y=0.97)

        fname = filename or f"comparison_spider_{'_'.join(p.split()[0].lower() for p in player_list)}.png"
        return _save_and_return(fig, fname)

    # ── CHART 6: Similarity Network ────────────────────────────────────────────
    def similarity_network(
        self,
        player_name: str,
        df: pd.DataFrame,
        n_similar: int = 8,
        distances: Optional[Dict[str, float]] = None,
        filename: Optional[str] = None,
    ) -> plt.Figure:
        """
        Network-style scatter: central player at centre, similar players in circle.

        Edge thickness is proportional to similarity (inverse distance).
        Node size is proportional to impact score.

        This visualises what scouts mean when they say "players in the same mould"
        — it's not opinion, it's Euclidean distance in composite-feature space.

        Parameters
        ----------
        player_name : str
        df : pd.DataFrame
        n_similar : int
            Number of similar players to show around the central node.
        distances : dict, optional
            {player_name: distance} for edge thickness. If None, edges are equal.
        """
        mask = df["player_name"].str.lower().str.contains(player_name.lower(), na=False)
        if not mask.any():
            logger.warning("Player '%s' not found for similarity network", player_name)
            return plt.figure()

        central_player = df[mask].iloc[0]
        central_name = central_player["player_name"]
        central_score = float(central_player.get("impact_score", 50))

        # Get similar player rows
        similar_names = []
        if "archetype_label" in df.columns:
            arch = central_player.get("archetype_label", "")
            same_arch = df[
                (df["archetype_label"] == arch)
                & (~df["player_name"].str.lower().str.contains(player_name.lower(), na=False))
            ]
            similar_names = same_arch.head(n_similar)["player_name"].tolist()

        if len(similar_names) < n_similar:
            pos = central_player.get("position", "")
            others = df[
                (df["position"] == pos)
                & (~df["player_name"].str.lower().str.contains(player_name.lower(), na=False))
                & (~df["player_name"].isin(similar_names))
            ].head(n_similar - len(similar_names))
            similar_names += others["player_name"].tolist()

        similar_names = similar_names[:n_similar]

        fig, ax = plt.subplots(figsize=(10, 10), facecolor=DARK_BG)
        ax.set_facecolor(DARK_BG)
        ax.set_aspect("equal")
        ax.axis("off")

        # Arrange similar players in circle
        theta = np.linspace(0, 2 * np.pi, len(similar_names), endpoint=False)
        radius = 3.5

        for i, sim_name in enumerate(similar_names):
            sim_mask = df["player_name"] == sim_name
            if not sim_mask.any():
                continue
            sim_player = df[sim_mask].iloc[0]
            sim_score = float(sim_player.get("impact_score", 50))

            sx = radius * np.cos(theta[i])
            sy = radius * np.sin(theta[i])

            # Edge
            dist = (distances or {}).get(sim_name, 1.0)
            similarity = 1.0 / (dist + 0.1)
            lw = min(4.0, similarity * 2.0)

            ax.plot([0, sx], [0, sy], color=GREEN, linewidth=lw, alpha=0.3, zorder=1)

            # Node size proportional to impact score
            node_size = max(30, sim_score * 4)
            ax.scatter(sx, sy, s=node_size, color=AMBER, zorder=3, edgecolors=WHITE,
                       linewidths=1.5)
            ax.text(sx, sy - 0.35, sim_name, ha="center", va="top",
                    color=WHITE, fontsize=7.5, fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=2, foreground=DARK_BG)])
            ax.text(sx, sy + 0.35, f"{sim_score:.0f}", ha="center", va="bottom",
                    color=AMBER, fontsize=7)

        # Central node
        central_size = max(100, central_score * 6)
        ax.scatter(0, 0, s=central_size, color=GREEN, zorder=4,
                   edgecolors=WHITE, linewidths=2.5)
        ax.text(0, -0.4, central_name, ha="center", va="top",
                color=GREEN, fontsize=11, fontweight="bold")
        ax.text(0, 0.45, f"{central_score:.0f}/100", ha="center", va="bottom",
                color=GREEN, fontsize=10)

        ax.set_xlim(-5.5, 5.5)
        ax.set_ylim(-5.5, 5.5)

        fig.suptitle(f"Similarity Network: {central_name}\n"
                     f"Players in the same mould (by composite-feature distance)",
                     color=WHITE, fontsize=12, fontweight="bold")

        fname = filename or f"similarity_network_{player_name.lower().replace(' ', '_')}.png"
        return _save_and_return(fig, fname)

    def generate_all_cm_charts(
        self,
        cm_df: pd.DataFrame,
        sample_player: str,
        comparison_players: List[str],
    ) -> None:
        """
        Generate all 6 chart types for the CM position as a demonstration.

        Parameters
        ----------
        cm_df : pd.DataFrame
        sample_player : str
        comparison_players : list of str
        """
        logger.info("Generating all charts for CM position")
        self.radar_chart(sample_player, cm_df, "cm_radar_sample_player.png")
        self.position_scatter(cm_df, "CM", "DAQ", "PPI", "cm_scatter_daq_vs_ppi.png")
        self.impact_score_distribution(cm_df, "CM", filename="cm_impact_distribution.png")
        self.archetype_profile_heatmap(cm_df, "CM", "cm_archetype_heatmap.png")
        self.player_comparison_spider(comparison_players, cm_df, "cm_comparison_spider.png")
        self.similarity_network(sample_player, cm_df, filename="cm_similarity_network.png")
        logger.info("All CM charts generated")
