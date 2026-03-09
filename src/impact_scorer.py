"""
impact_scorer.py
================
The core rating engine. Converts composite metrics into a single 0-100
impact score per player, normalised within their position group.

PHILOSOPHY:
A player's impact score answers one question: "How much does this player
help their team win football games, relative to other players in the same
role?" It is explicitly NOT a cross-position ranking — a striker who scores
73/100 is not directly comparable to a centre-back who scores 73/100. They
are measured on different actions because they do different jobs.

The scoring system is a weighted sum of normalised composite metrics.
Weights ARE the football opinion baked into the system. Every weight has
an explanation. The system is fully auditable: a scout can point to
"high DAQ but weak BRS — dominant defender who loses the ball too often"
and have a conversation about it. A neural network cannot provide this.

VALIDATION METHODOLOGY (for future real-data implementation):
To validate that impact_score correlates with actual player value:
1. Correlate with end-of-season Transfermarkt market value change
   (methodology: Herm et al., 2014; Müller et al., 2017)
2. Correlate with manager selection frequency — players with high scores
   who rarely play may indicate injury issues or context (Pappalardo et al., 2019)
3. Team performance delta: compare team xG difference in games with vs without
   the player — this is the gold-standard VAEP approach (Decroos et al., 2019)
4. Cross-validate against established metrics: Opta's rankings, StatsBomb VAEP
   scores where publicly available. Agreement across systems gives confidence.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# ── Position weight definitions ───────────────────────────────────────────────
# These weights encode the football philosophy of what each position is FOR.
# Changing these weights is the primary mechanism for tuning the system.
# Each weight set sums to 1.0.

CB_WEIGHTS: Dict[str, float] = {
    "DAQ_normalized": 0.35,   # defending is the CB's primary job — the weight majority here
    "BRS_normalized": 0.25,   # ball-playing CBs are the premium asset in modern football
                               # (Virgil van Dijk, Ruben Dias command fees because of this)
    "PPI_normalized": 0.20,   # progressive carrying from deep is a major tactical asset
                               # (van Dijk's 50m carries to break lines; Laporte's switch passes)
    "PII_normalized": 0.10,   # CBs press in high-line systems (Arsenal CBs press aggressively)
    "CCC_normalized": 0.10,   # rare but high-value when a CB arrives late in the box
}

FB_WEIGHTS: Dict[str, float] = {
    "PPI_normalized": 0.30,   # overlapping/underlapping FB — Trent TAA, Robertson, Cancelo
                               # — is defined by how much they progress the ball
    "CCC_normalized": 0.25,   # modern FBs are primary attacking contributors — Trent in 21-22
                               # created more xAG than many 10s; this reflects that evolution
    "DAQ_normalized": 0.25,   # FBs still need to defend (even Trent's defensive issues
                               # show up here — it's weighted to not dismiss the job requirement)
    "BRS_normalized": 0.15,   # ball retention when receiving high/wide is important
    "PII_normalized": 0.05,   # FBs press less — wing coverage is their shape responsibility
}

CM_WEIGHTS: Dict[str, float] = {
    "PPI_normalized": 0.30,   # the CM is the engine of ball progression — the most important
                               # metric for the modern 8 or 6 who controls tempo
    "BRS_normalized": 0.25,   # you cannot progress what you give away — CMs who turn over
                               # ball in transition are a liability regardless of pass volume
    "CCC_normalized": 0.20,   # the creative 8 (De Bruyne-lite, Eriksen) creates from deep
    "DAQ_normalized": 0.15,   # the pressing CM (Kanté, Kovacic) contributes here
    "PII_normalized": 0.10,   # effective pressing from midfield — crucial in high-block breaks
}

ST_WEIGHTS: Dict[str, float] = {
    "TGI_normalized": 0.45,   # primary job of a striker is to generate and convert goal threat
                               # — this is the single largest weight across any position
    "PPI_normalized": 0.20,   # link play and hold-up — Bobby Firmino defined this profile;
                               # a striker who can progress possession is a tactical multiplier
    "PII_normalized": 0.20,   # the pressing striker is the first line of defensive engagement
                               # — Firmino/Salah's press output was as important as their goals
    "BRS_normalized": 0.10,   # retaining the ball under physical pressure in the final third
    "CCC_normalized": 0.05,   # assists from strikers are a bonus, not a job requirement
                               # — Haaland rarely assists; this shouldn't punish him much
}

GK_WEIGHTS: Dict[str, float] = {
    "GK_SHOT_STOPPING_normalized": 0.55,         # primary job: stop shots
    "GK_DISTRIBUTION_QUALITY_normalized": 0.25,  # modern GK must play from the back
    "GK_SWEEPER_KEEPER_INDEX_normalized": 0.20,  # high-line support — Ederson/Alisson profile
}

POSITION_WEIGHTS: Dict[str, Dict[str, float]] = {
    "CB": CB_WEIGHTS,
    "FB": FB_WEIGHTS,
    "CM": CM_WEIGHTS,
    "ST": ST_WEIGHTS,
    "GK": GK_WEIGHTS,
}

POSITION_COMPONENTS: Dict[str, List[str]] = {
    "CB": ["DAQ", "BRS", "PPI", "PII", "CCC"],
    "FB": ["PPI", "CCC", "DAQ", "BRS", "PII"],
    "CM": ["PPI", "BRS", "CCC", "DAQ", "PII"],
    "ST": ["TGI", "PPI", "PII", "BRS", "CCC"],
    "GK": ["GK_SHOT_STOPPING", "GK_DISTRIBUTION_QUALITY", "GK_SWEEPER_KEEPER_INDEX"],
}


class ImpactScorer:
    """
    Computes the final 0-100 impact score for each player within their position group.

    The two-stage normalisation approach is a deliberate design choice:
    1. MinMaxScaler within each position group (not globally) ensures that
       a CB's defending score is relative to other CBs, not to strikers who
       barely tackle. Cross-position normalisation would compress all strikers
       to near-zero on defensive metrics, destroying meaningful variation.
    2. The weighted combination allows position-specific philosophical tuning.
       Adjusting weights is the analyst's primary lever for reflecting tactical
       evolution (e.g., as FBs become more attacking, increase CCC weight).
    """

    def normalize_within_position(
        self, df: pd.DataFrame, columns: List[str]
    ) -> pd.DataFrame:
        """
        Apply MinMaxScaler to specified columns, adding _normalized suffix.

        MinMaxScaler chosen for the scoring layer (not StandardScaler) because:
        - Produces bounded 0-100 output that humans can interpret immediately
        - Preserves monotonic ordering (rank order unchanged after scaling)
        - The winsorisation in preprocessing ensures extreme values don't
          collapse all other players to near-zero (which MinMax would do
          without winsorisation)
        - A scout immediately understands "93/100 on DAQ" — they do not
          understand "+2.7 standard deviations above mean on DAQ"

        Parameters
        ----------
        df : pd.DataFrame
            Position-specific DataFrame with composite feature columns.
        columns : list of str
            Columns to normalise.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with additional {col}_normalized columns scaled 0–1.
        """
        df = df.copy()
        scaler = MinMaxScaler()

        valid_cols = [c for c in columns if c in df.columns]
        if not valid_cols:
            logger.warning("No valid columns to normalize in: %s", columns)
            return df

        # Fill NaN with column median before scaling (defensive programming)
        df[valid_cols] = df[valid_cols].fillna(df[valid_cols].median())
        scaled = scaler.fit_transform(df[valid_cols])

        for i, col in enumerate(valid_cols):
            df[f"{col}_normalized"] = scaled[:, i]

        return df

    def calculate_position_weights(self) -> Dict[str, Dict[str, float]]:
        """
        Return the position weight dictionaries.

        These weights are the football opinion baked into the system.
        They are hardcoded here to be explicitly auditable — if a weight
        needs to change to reflect tactical evolution (e.g., FBs becoming
        even more offensive in 2025-26), this is the single place to update it.

        Returns
        -------
        dict
            Position → {component_normalized: weight} mapping.
        """
        return POSITION_WEIGHTS

    def compute_impact_score(
        self, df: pd.DataFrame, position: str
    ) -> pd.Series:
        """
        Compute the final weighted impact score for all players in a position.

        Steps:
        1. Normalise all component scores to 0-1 within the position group
        2. Apply position-specific weights to compute weighted sum
        3. Scale to 0-100 for readability

        The result is the player's impact score relative to their peers in
        the same position. The highest scorer gets 100, the lowest gets 0.
        The middle of the distribution is around 40-60 for most positions.

        Parameters
        ----------
        df : pd.DataFrame
            Position-specific DataFrame with composite features (PPI, DAQ, etc.)
        position : str
            One of 'GK', 'CB', 'FB', 'CM', 'ST'

        Returns
        -------
        pd.Series
            'impact_score' scaled 0-100. Same index as input df.
        """
        components = POSITION_COMPONENTS[position]
        weights = POSITION_WEIGHTS[position]

        df_norm = self.normalize_within_position(df, components)

        score = pd.Series(0.0, index=df_norm.index)
        for col, weight in weights.items():
            if col in df_norm.columns:
                score += df_norm[col] * weight
            else:
                logger.warning("Weight column %s not found in DataFrame", col)

        # Scale to 0-100
        score_scaled = score * 100
        return score_scaled.rename("impact_score")

    def generate_player_card(
        self, player_name: str, df: pd.DataFrame
    ) -> Dict:
        """
        Generate a comprehensive player card dictionary for a named player.

        The player card is the primary output for presenting individual
        player assessments. It mirrors what an analyst would include in
        a scout report: position context, component breakdown, relative
        standing, and a comparable archetype label.

        Parameters
        ----------
        player_name : str
            Player to look up. Case-insensitive substring match allowed.
        df : pd.DataFrame
            Full scored and clustered DataFrame (should include 'impact_score',
            all component scores, and 'archetype_label' if available).

        Returns
        -------
        dict
            Keys: name, club, position, age, impact_score, component_scores,
            percentile_in_position, top_strength, biggest_weakness,
            comparable_archetype
        """
        # Case-insensitive lookup
        mask = df["player_name"].str.lower().str.contains(player_name.lower(), na=False)
        if not mask.any():
            logger.warning("Player '%s' not found in DataFrame", player_name)
            return {"error": f"Player '{player_name}' not found"}

        player = df[mask].iloc[0]
        position = player.get("position", "Unknown")

        # Component scores for this position
        components = POSITION_COMPONENTS.get(position, [])
        component_scores = {}
        for comp in components:
            if comp in df.columns:
                component_scores[comp] = round(float(player[comp]), 3)

        # Percentile within position
        pos_df = df[df["position"] == position]
        if "impact_score" in df.columns and len(pos_df) > 0:
            percentile = float(
                (pos_df["impact_score"] <= player["impact_score"]).mean() * 100
            )
        else:
            percentile = float("nan")

        # Strength and weakness from normalised components
        norm_components = {k: v for k, v in player.items()
                           if "_normalized" in str(k) and pd.notna(v)}
        if norm_components:
            top_strength = max(norm_components, key=norm_components.get).replace("_normalized", "")
            biggest_weakness = min(norm_components, key=norm_components.get).replace("_normalized", "")
        else:
            top_strength = "N/A"
            biggest_weakness = "N/A"

        archetype = player.get("archetype_label", "Not clustered")

        card = {
            "name": player["player_name"],
            "club": player.get("club", "Unknown"),
            "position": position,
            "age": int(player.get("age", 0)),
            "minutes_played": int(player.get("minutes_played", 0)),
            "impact_score": round(float(player.get("impact_score", 0)), 1),
            "component_scores": component_scores,
            "percentile_in_position": round(percentile, 1),
            "top_strength": top_strength,
            "biggest_weakness": biggest_weakness,
            "comparable_archetype": archetype,
        }
        return card

    def compare_players(
        self, player_list: List[str], df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Side-by-side component breakdown for a list of named players.

        Cross-position comparison WARNING:
        When players from different positions appear in the same comparison,
        this method logs a prominent warning because cross-position comparison
        is methodologically weak. A striker's PPI score of 65 is relative to
        other strikers; a CM's PPI of 65 is relative to other CMs — who are
        much more active progressors on average. Comparing the two numbers
        is like comparing a winger's defensive tackles to a CDM's: the role
        context makes the raw number meaningless.

        For valid cross-position comparison, use absolute raw stats (per 90),
        not position-normalised scores.

        Parameters
        ----------
        player_list : list of str
            Player names to compare.
        df : pd.DataFrame
            Scored and clustered DataFrame.

        Returns
        -------
        pd.DataFrame
            Players as rows, component scores as columns. Includes a note
            column flagging cross-position comparisons.
        """
        rows = []
        positions_seen = set()

        for name in player_list:
            mask = df["player_name"].str.lower().str.contains(name.lower(), na=False)
            if not mask.any():
                logger.warning("Player '%s' not found for comparison", name)
                continue
            player = df[mask].iloc[0].copy()
            positions_seen.add(player.get("position", "Unknown"))
            rows.append(player)

        if not rows:
            return pd.DataFrame()

        if len(positions_seen) > 1:
            positions_str = ", ".join(sorted(positions_seen))
            warning_msg = (
                f"\n{'='*70}\n"
                f"⚠  CROSS-POSITION COMPARISON WARNING\n"
                f"{'='*70}\n"
                f"You are comparing players from: {positions_str}\n\n"
                f"Impact scores and component scores are normalised WITHIN each\n"
                f"position group. A CB scoring 75/100 on DAQ and a CM scoring\n"
                f"75/100 on DAQ have NOT performed the same defensive actions —\n"
                f"the CB is in the top 25% of CBs on defending, the CM in the\n"
                f"top 25% of CMs. CBs average ~2x the defensive actions of CMs.\n\n"
                f"Cross-position score comparison is like comparing a winger's\n"
                f"tackles to a CDM's: the role context makes the number meaningless.\n"
                f"Use raw per-90 stats for cross-position comparisons.\n"
                f"{'='*70}\n"
            )
            logger.warning(warning_msg)
            print(warning_msg)

        result_df = pd.DataFrame(rows)

        # Select display columns
        display_cols = ["player_name", "club", "position", "age", "impact_score"]
        for comp in ["PPI", "DAQ", "CCC", "BRS", "PII", "TGI",
                     "GK_SHOT_STOPPING", "GK_DISTRIBUTION_QUALITY", "GK_SWEEPER_KEEPER_INDEX"]:
            if comp in result_df.columns:
                display_cols.append(comp)

        available = [c for c in display_cols if c in result_df.columns]
        return result_df[available].set_index("player_name")

    def score_all_positions(
        self, position_dfs: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Apply impact scoring to all position DataFrames.

        Parameters
        ----------
        position_dfs : dict
            Feature-engineered position DataFrames.

        Returns
        -------
        dict
            Same structure with 'impact_score' column added and
            all _normalized columns preserved for later use.
        """
        scored = {}
        for pos, df in position_dfs.items():
            logger.info("Scoring %s players (%d)", pos, len(df))
            df_norm = self.normalize_within_position(df, POSITION_COMPONENTS[pos])
            df_norm["impact_score"] = self.compute_impact_score(df_norm, pos)
            scored[pos] = df_norm
            top = df_norm.nlargest(3, "impact_score")[["player_name", "impact_score"]].values
            logger.info("Top 3 %s: %s", pos, top)
        return scored
