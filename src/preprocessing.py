"""
preprocessing.py
================
Data cleaning and preparation for the football impact rating pipeline.

Football analytics has a specific data quality challenge: the minimum minutes
threshold is not a technical artefact but a genuine analytical boundary.
A player who has played 120 minutes across 3 substitute appearances does not
have a per-90 profile — they have 3 data points that will produce extreme
per-90 numbers due to sample variance. The 900-minute threshold (~10 full
games) is the FBref standard and has become an analytics community norm.

Outlier handling is the most philosophically important decision here:
we WINSORISE rather than remove because in football, outliers ARE the signal.
A striker averaging 0.72 xG/90 over a season (Erling Haaland in 22-23) is not
a data error — it is the most important data point in the dataset. Removing it
because it sits 4 standard deviations from the mean would be analytically
illiterate. We cap values to prevent them from mathematically dominating
scaled outputs while preserving their status as the highest-scoring values.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Outfield stat columns (GK-specific columns excluded) ─────────────────────
OUTFIELD_STAT_COLS = [
    "goals", "assists", "xG", "xAG",
    "progressive_carries", "progressive_passes", "progressive_passes_received",
    "pressures", "pressure_success_rate",
    "tackles_won", "interceptions",
    "aerial_duels_won_pct", "aerials_attempted", "ground_duels_won_pct",
    "pass_completion_pct", "key_passes", "passes_into_final_third",
    "shot_creating_actions", "goal_creating_actions",
    "blocks", "clearances", "errors_leading_to_shot",
    "carries_into_final_third", "miscontrols", "dispossessed",
]

GK_STAT_COLS = [
    "PSxG_minus_GA", "save_pct", "passes_launched_pct",
    "avg_pass_length", "gk_sweeper_actions",
]

# ── Columns used for each position subset ─────────────────────────────────────
POSITION_COLS: Dict[str, list] = {
    "GK": ["player_name", "club", "position", "age", "minutes_played"] + GK_STAT_COLS,
    "CB": ["player_name", "club", "position", "age", "minutes_played"] + OUTFIELD_STAT_COLS,
    "FB": ["player_name", "club", "position", "age", "minutes_played"] + OUTFIELD_STAT_COLS,
    "CM": ["player_name", "club", "position", "age", "minutes_played"] + OUTFIELD_STAT_COLS,
    "ST": ["player_name", "club", "position", "age", "minutes_played"] + OUTFIELD_STAT_COLS,
}


class DataPreprocessor:
    """
    Cleans, filters, and validates the raw player dataset.

    The three-stage pipeline mirrors the standard football analytics workflow:
    1. Minimum minutes filter — removes statistically unreliable samples
    2. Position-aware winsorisation — preserves outliers as signal, not noise
    3. Separation by position — ensures downstream metrics are position-relative
    """

    def filter_minimum_minutes(
        self, df: pd.DataFrame, threshold: int = 900
    ) -> pd.DataFrame:
        """
        Remove players with fewer than `threshold` minutes played.

        Football reason: Per-90 stats become meaningless below ~10 full games.
        A player who scored in 2 of his 3 sub appearances has a 0.67 goals/90
        rate that tells us nothing about his true ability — only his luck in
        small samples. 900 minutes (~10 full games) is the FBref community
        standard. Some analysts use 1080 (12 games) for a stricter cut.

        Parameters
        ----------
        df : pd.DataFrame
            Full player DataFrame with 'minutes_played' column.
        threshold : int
            Minimum minutes. Default 900 (FBref standard).

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame. Players below threshold are dropped entirely.
        """
        before = len(df)
        df_filtered = df[df["minutes_played"] >= threshold].copy()
        after = len(df_filtered)
        removed = before - after
        logger.info(
            "Minutes filter (%d mins): %d players removed, %d retained",
            threshold, removed, after
        )
        return df_filtered.reset_index(drop=True)

    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Winsorise numeric columns using the IQR method, applied WITHIN each
        position group.

        WHY WINSORISE, NOT REMOVE:
        In football analytics, outliers are almost never data errors — they are
        the most interesting players. Erling Haaland's 0.72 xG/90 in 2022-23
        is not noise; it is the central finding. Removing it would leave us with
        a dataset that cannot distinguish elite from good players.

        Winsorisation (capping at Q1 - 1.5*IQR and Q3 + 1.5*IQR) preserves the
        outlier's rank-ordering (it remains the maximum value) while preventing
        it from dominating when we later apply Min-Max scaling. Without capping,
        a single 3-sigma outlier compresses all other players into the bottom
        10% of a 0-100 scale, destroying discrimination across the majority.

        WHY PER-POSITION:
        A striker's 3.1 progressive carries/90 is high for a striker but below
        average for a full-back. Applying IQR across all positions would incorrectly
        flag the striker as an outlier in the wrong direction.

        Parameters
        ----------
        df : pd.DataFrame
            Player DataFrame after minutes filtering.

        Returns
        -------
        pd.DataFrame
            Winsorised DataFrame. Shape unchanged.
        """
        df_out = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = ["age", "minutes_played"]
        stat_cols = [c for c in numeric_cols if c not in exclude]

        for position in df["position"].unique():
            mask = df_out["position"] == position
            pos_df = df_out.loc[mask, stat_cols]

            for col in stat_cols:
                series = pos_df[col].dropna()
                if len(series) < 4:
                    continue
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                df_out.loc[mask, col] = df_out.loc[mask, col].clip(lower=lower, upper=upper)

        logger.info("Winsorisation complete (IQR per position group)")
        return df_out

    def separate_by_position(
        self, df: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        Split the full DataFrame into position-specific subsets, retaining
        only columns relevant to each position.

        GKs are evaluated on fundamentally different metrics (save percentage,
        distribution quality) so mixing them with outfield players for any
        metric calculation would be analytically invalid.

        Parameters
        ----------
        df : pd.DataFrame
            Full cleaned DataFrame.

        Returns
        -------
        dict
            Keys: ['GK', 'CB', 'FB', 'CM', 'ST']
            Values: Position-specific DataFrames with relevant columns only.
        """
        result: Dict[str, pd.DataFrame] = {}
        for pos, cols in POSITION_COLS.items():
            available_cols = [c for c in cols if c in df.columns]
            pos_df = df[df["position"] == pos][available_cols].copy()
            pos_df = pos_df.reset_index(drop=True)
            result[pos] = pos_df
            logger.info("Position %s: %d players, %d columns", pos, len(pos_df), len(pos_df.columns))
        return result

    def validate_per90(self, df: pd.DataFrame) -> None:
        """
        Assert that numeric columns are within analytically valid bounds.

        These are not arbitrary — they reflect hard football knowledge:
        - xG cannot exceed 2.0 per 90: no player in history has averaged >1.0
          over a full season; the synthetic cap at 1.5 gives headroom
        - Completion rates must be 0-100%: they are percentages
        - Aerial win % must be 0-100%: same reason

        Raises ValueError with a descriptive message if any check fails.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate.
        """
        errors = []

        if "xG" in df.columns:
            bad_xg = (df["xG"].dropna() > 2.0).sum()
            if bad_xg > 0:
                errors.append(
                    f"xG: {bad_xg} players have xG > 2.0 per 90. "
                    "No player in football history has sustained this rate. "
                    "Check data generation parameters."
                )

        for pct_col in ["pass_completion_pct", "pressure_success_rate",
                        "aerial_duels_won_pct", "ground_duels_won_pct"]:
            if pct_col in df.columns:
                out_of_range = (
                    (df[pct_col].dropna() < 0) | (df[pct_col].dropna() > 100)
                ).sum()
                if out_of_range > 0:
                    errors.append(
                        f"{pct_col}: {out_of_range} values outside 0-100 range. "
                        "Percentage column contains impossible values."
                    )

        if errors:
            raise ValueError("Data validation failed:\n" + "\n".join(errors))

        logger.info("Validation passed. Dataset shape: %s", df.shape)
        logger.info("Numeric summary:\n%s", df.describe().to_string())

    def run(
        self, df: pd.DataFrame, min_minutes: int = 900
    ) -> pd.DataFrame:
        """
        Full preprocessing pipeline: filter → winsorise → validate.

        Parameters
        ----------
        df : pd.DataFrame
            Raw player DataFrame from data_generator.
        min_minutes : int
            Minimum minutes threshold. Default 900.

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame ready for feature engineering.
        """
        logger.info("Starting preprocessing pipeline")
        df = self.filter_minimum_minutes(df, min_minutes)
        df = self.handle_outliers(df)
        self.validate_per90(df)
        logger.info("Preprocessing complete. Final shape: %s", df.shape)
        return df
