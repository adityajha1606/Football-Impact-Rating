"""
feature_engineering.py
======================
Constructs composite performance metrics from raw per-90 statistics.

This is the analytical heart of the system. Raw per-90 stats are noisy,
correlated, and position-dependent. Composite metrics aggregate related
actions into dimensions that map onto real football concepts: a player
either progresses the ball, defends well, creates chances, retains
possession under pressure, or presses with purpose. These five dimensions
(plus a sixth for pure threat generation) capture everything a scout would
articulate about an outfield player's contribution.

Every weight in every formula has a football reason. Where the weight is
not obvious, the docstring includes the specific analytical reasoning and
a real-world player example who represents the extreme of that dimension.

GKs are handled entirely separately — their contribution dimensions are
fundamentally different and cannot be folded into the outfield framework
without producing nonsensical results.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Builds composite metrics from raw per-90 statistics.

    All composite features are designed to be:
    - Directional: higher is better (except for penalty stats like miscontrols)
    - Interpretable: each component can be explained to a non-technical audience
    - Position-sensitive: used differently per position in the scoring layer
    - Robust: designed to reduce the variance of individual raw stats
    """

    # ── Safe division helper ──────────────────────────────────────────────────
    @staticmethod
    def _safe_div(a: pd.Series, b: pd.Series, fill: float = 0.0) -> pd.Series:
        """Divide series avoiding division by zero."""
        return a.div(b.replace(0, np.nan)).fillna(fill)

    # ══════════════════════════════════════════════════════════════════════════
    # COMPOSITE METRIC 1: POSSESSION PROGRESSION INDEX (PPI)
    # ══════════════════════════════════════════════════════════════════════════
    def possession_progression_index(self, df: pd.DataFrame) -> pd.Series:
        """
        Possession Progression Index (PPI) — measures how much a player
        advances the ball up the pitch per 90 minutes.

        Formula:
            PPI = (progressive_carries * 1.0
                 + progressive_passes * 0.7
                 + progressive_passes_received * 0.4
                 + passes_into_final_third * 0.8) / 4

        Weight rationale:
        - progressive_carries (1.0): A carry is a single player advancing
          the ball at personal risk — it signals both technical ability AND
          decision-making. Trent Alexander-Arnold's 6.1 prog carries/90 in
          23-24 is the defining feature of his attacking profile.
        - progressive_passes (0.7): Passes advance the ball but distribute
          credit between passer and receiver. Still highly valuable but slightly
          discounted vs carries.
        - progressive_passes_received (0.4): Shows good positioning and off-ball
          movement but is partly teammate-dependent. A striker receiving 8
          progressive passes/90 benefits from his fullback's delivery. Discounted
          to avoid over-crediting players on possession-dominant teams.
        - passes_into_final_third (0.8): High-value destination passes that
          directly create danger — weighted above normal prog passes but below
          carries due to the quality threshold implied.

        Real-world anchor: Kevin De Bruyne's PPI would be elite via progressive
        passes and passes_into_final_third. Declan Rice's PPI is elite via
        progressive carries combined with progressive passes.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain: progressive_carries, progressive_passes,
            progressive_passes_received, passes_into_final_third
        """
        ppi = (
            df["progressive_carries"] * 1.0          # carries: highest risk/reward
            + df["progressive_passes"] * 0.7         # passes: effective but shared credit
            + df["progressive_passes_received"] * 0.4  # receiving: positioning proxy, team-dependent
            + df["passes_into_final_third"] * 0.8    # final-third entries: high danger value
        ) / 4
        return ppi.rename("PPI")

    # ══════════════════════════════════════════════════════════════════════════
    # COMPOSITE METRIC 2: DEFENSIVE ACTION QUALITY (DAQ)
    # ══════════════════════════════════════════════════════════════════════════
    def defensive_action_quality(self, df: pd.DataFrame) -> pd.Series:
        """
        Defensive Action Quality (DAQ) — measures the volume AND effectiveness
        of defensive contributions per 90.

        Formula:
            DAQ = (tackles_won * 1.2
                 + interceptions * 1.0
                 + pressures * 0.3 * (pressure_success_rate / 100)
                 + (aerial_duels_won_pct / 100) * aerials_attempted * 0.8) / 4

        Weight rationale:
        - tackles_won (1.2): The most direct defensive action — physically
          winning the ball from an opponent. Weighted highest because it
          requires both technical and physical ability. Only WON tackles counted
          (tackles attempted would include failed challenges, which are negative).
        - interceptions (1.0): Anticipatory defending. Shows reading of the game
          and positional intelligence. No success/failure split available so
          counted as-is (by definition an interception is a success).
        - pressures * success_rate (0.3 with multiplication): Volume of pressing
          without success rate is noise. Burnley under Sean Dyche in 2022-23 had
          high press volume (desperate defending in their own half) but extremely
          low success rates. The multiplication ensures only EFFECTIVE pressing
          contributes. 0.3 base weight because even successful pressing is a
          lower-value action than winning a tackle.
        - aerial_duels_won_pct * attempted (0.8): Win percentage alone is
          misleading — a player who wins 80% of 1 aerial per game is less
          dominant than one who wins 60% of 8. Multiplying by attempts gives
          "aerial duels actually won", then weight 0.8 reflects the importance
          of aerial dominance especially for CBs and physical strikers.

        Real-world anchor: N'Golo Kanté would score elite on tackles_won +
        interceptions + effective pressing. A vintage John Terry would score
        high via aerials + tackles. Rúben Dias scores high via tackles + aerials.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain: tackles_won, interceptions, pressures,
            pressure_success_rate, aerial_duels_won_pct, aerials_attempted
        """
        effective_pressing = (
            df["pressures"]
            * (df["pressure_success_rate"] / 100)  # normalise pct to fraction
            * 0.3  # 0.3: pressing weighted lower — volume without success inflates for low-block teams
        )
        aerial_contribution = (
            (df["aerial_duels_won_pct"] / 100) * df["aerials_attempted"] * 0.8
        )
        daq = (
            df["tackles_won"] * 1.2      # 1.2: winning physical duels is the primary defensive act
            + df["interceptions"] * 1.0  # 1.0: reading the game — equally valuable
            + effective_pressing
            + aerial_contribution
        ) / 4
        return daq.rename("DAQ")

    # ══════════════════════════════════════════════════════════════════════════
    # COMPOSITE METRIC 3: CHANCE CREATION CONTRIBUTION (CCC)
    # ══════════════════════════════════════════════════════════════════════════
    def chance_creation_contribution(self, df: pd.DataFrame) -> pd.Series:
        """
        Chance Creation Contribution (CCC) — measures both the volume and
        quality of chances created per 90.

        Formula:
            CCC = (key_passes * 1.0
                 + shot_creating_actions * 0.6
                 + goal_creating_actions * 1.5
                 + xAG * 2.0) / 4

        Weight rationale:
        - key_passes (1.0): Traditional metric — passes that directly lead to
          a shot. Good signal but blunt: a cutback to a player who shoots
          straight at the keeper counts equally to a through-ball for a
          one-on-one. Hence not the highest weight.
        - shot_creating_actions (0.6): Broader definition includes dribbles,
          fouls won, and secondary passes. High volume signal but lower quality
          per unit. Discounted to avoid inflating for high-touch players.
        - goal_creating_actions (1.5): The two-action sequence that directly
          leads to a goal. Much rarer but far more outcome-valuable. Premium
          weight to reward players who operate in the most dangerous phases.
        - xAG * 2.0: Expected Assisted Goals — the cleanest single measure of
          chance quality created. Unlike key passes (binary) or SCA (broad),
          xAG incorporates shot difficulty and location to measure the actual
          goal probability generated. De Bruyne's xAG is consistently the PL's
          highest because his assists are high-quality chances, not tap-ins from
          his cutbacks. Double weight to reflect its analytical superiority.

        Real-world anchor: Kevin De Bruyne's CCC would be the highest in the
        dataset via xAG and goal_creating_actions. Assists-heavy wide players
        who create low-xG shots would score lower than their assist tally suggests.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain: key_passes, shot_creating_actions,
            goal_creating_actions, xAG
        """
        ccc = (
            df["key_passes"] * 1.0              # 1.0: standard chance creation metric
            + df["shot_creating_actions"] * 0.6  # 0.6: broad volume measure, discounted
            + df["goal_creating_actions"] * 1.5  # 1.5: rare but high-value direct contributions
            + df["xAG"] * 2.0                    # 2.0: best quality-adjusted chance creation metric
        ) / 4
        return ccc.rename("CCC")

    # ══════════════════════════════════════════════════════════════════════════
    # COMPOSITE METRIC 4: BALL RETENTION SCORE (BRS)
    # ══════════════════════════════════════════════════════════════════════════
    def ball_retention_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Ball Retention Score (BRS) — measures how well a player maintains
        possession under pressure and advances it smartly.

        Formula:
            BRS = (pass_completion_pct / 100 * 2.0)
                  - (miscontrols * 0.3)
                  - (dispossessed * 0.4)
                  + (carries_into_final_third * 0.5)

        Weight rationale:
        - pass_completion_pct / 100 * 2.0: Normalised completion rate is the
          foundation of ball retention. Divided by 100 to express as a fraction,
          then *2.0 to give it appropriate scale in the formula. However, raw
          completion rate rewards players who only attempt short safe passes
          (think a deep-lying CB playing 20-yard laterals). Hence the penalties.
        - miscontrols * -0.3: Technical errors — the ball deflected off a poor
          first touch. These are low-event but high-impact: one miscontrol in a
          dangerous area can be catastrophic. 0.3 per incident is modest because
          high-volume touch players will naturally have more absolute miscontrols.
        - dispossessed * -0.4: Losing the ball while being challenged is the most
          serious ball-retention failure (vs a miscontrol which may not lose
          possession). Higher penalty than miscontrol. Strikers who carry into
          dangerous areas will have higher dispossession rates — this is INTENDED;
          they should be capturing that value through TGI's carries_into_ft bonus.
        - carries_into_final_third * 0.5: Smart progressors who carry effectively
          deserve a bonus because their high dispossession rate is acceptance of
          risk in high-value zones. Separates risk-accepters from timid holders.

        Real-world anchor: Sergio Busquets has a near-perfect BRS — elite
        completion rate, minimal miscontrols, almost never dispossessed. A
        Wilfried Zaha-style dribbler would have high BRS via carries bonus
        despite high dispossession, reflecting the valuable tradeoff they make.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain: pass_completion_pct, miscontrols,
            dispossessed, carries_into_final_third
        """
        brs = (
            (df["pass_completion_pct"] / 100) * 2.0  # normalise pct; 2.0 gives appropriate scale
            - df["miscontrols"] * 0.3                 # 0.3: modest penalty — absolute count favors volume
            - df["dispossessed"] * 0.4                # 0.4: losing possession under challenge, serious
            + df["carries_into_final_third"] * 0.5    # 0.5: bonus for productive risk-taking
        )
        return brs.rename("BRS")

    # ══════════════════════════════════════════════════════════════════════════
    # COMPOSITE METRIC 5: PRESSING INTENSITY INDEX (PII)
    # ══════════════════════════════════════════════════════════════════════════
    def pressing_intensity_index(self, df: pd.DataFrame) -> pd.Series:
        """
        Pressing Intensity Index (PII) — measures the effective output of a
        player's pressing work, not just the volume.

        Formula:
            PII = pressures
                  * (pressure_success_rate / 100)
                  * (1 + carries_into_final_third * 0.1)

        Weight rationale:
        The multiplicative structure is deliberate. A player who presses 40
        times per 90 but wins the ball back only 15% of the time generates
        6.0 effective press recoveries — the same as a player who presses 20
        times at 30% success rate. Volume and quality are both necessary.

        The progressive carry modifier (1 + carries_ftd * 0.1) captures the
        "press-and-go" player archetype: players who not only win the ball
        high but immediately drive forward, compounding the territorial gain
        from the press. This distinguishes Pressing intensity with forward
        threat (Musiala/Bernardo Silva profile) from high-press-then-recycle
        players (Fabinho pressing down but distributing laterally).

        The +1 base ensures the multiplier never collapses PII to zero for
        players with limited carries_into_ftd; it's a bonus, not a requirement.

        Real-world anchor: Without this metric, Burnley under Dyche (2022-23)
        would show some of the highest "pressing" numbers in the PL — they
        pressed desperately at 4-4-2 but recovered the ball infrequently.
        PII correctly identifies them as low-value pressers, while Man City
        (high success rate, immediate progression) score elite.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain: pressures, pressure_success_rate,
            carries_into_final_third
        """
        carry_bonus = 1 + df["carries_into_final_third"] * 0.1  # 0.1: modest bonus for press-and-go
        pii = (
            df["pressures"]
            * (df["pressure_success_rate"] / 100)  # normalise pct to fraction
            * carry_bonus
        )
        return pii.rename("PII")

    # ══════════════════════════════════════════════════════════════════════════
    # COMPOSITE METRIC 6: THREAT GENERATION INDEX (TGI) — attackers only
    # ══════════════════════════════════════════════════════════════════════════
    def threat_generation_index(self, df: pd.DataFrame) -> pd.Series:
        """
        Threat Generation Index (TGI) — measures net goal threat created AND
        converted by a player per 90, penalising defensive errors.

        Designed for ST, CF, and advanced midfielders. Applying to CBs or GKs
        produces meaningless results due to near-zero xG values.

        Formula:
            TGI = xG * 2.0
                  + xAG * 1.5
                  + shot_creating_actions * 0.4
                  + goal_creating_actions * 0.8
                  - errors_leading_to_shot * 1.5

        Weight rationale:
        - xG * 2.0: Expected goals is the primary output metric for attacking
          players. Double weight to ensure threat conversion dominates the index
          for pure strikers. Haaland's 0.72 xG/90 in 22-23 would be the
          overwhelming driver of his TGI.
        - xAG * 1.5: Creating chances for others is valuable for number 10s and
          wide players. Discounted vs xG because the striker's primary job is
          converting, not creating.
        - shot_creating_actions * 0.4: Volume signal — aggressive, involved
          attackers create more shots. Low individual weight because SCA is a
          broad category that includes very low-quality actions.
        - goal_creating_actions * 0.8: The two-action sequences that directly
          led to goals. Moderate weight; rarer than SCA but highly outcome-relevant.
        - errors_leading_to_shot * -1.5: The penalty for defensive mistakes is
          analytically critical. A striker who generates 0.40 xG but makes an
          error allowing a 0.25 xG counter-attack chance is net contributing
          only 0.15 xG on those actions. Attackers who press poorly and turn
          over the ball in dangerous areas should be penalised — this is the
          only metric that captures it. Penalty weighted at 1.5 (higher than
          xAG) to reflect the asymmetric cost of conceding from own errors.

        Real-world anchor: A clinical striker like Mo Salah (high xG, low errors)
        will vastly outscore a streaky wide player who creates lots of shots
        (high SCA) but frequently turns over near his own box.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain: xG, xAG, shot_creating_actions,
            goal_creating_actions, errors_leading_to_shot
        """
        tgi = (
            df["xG"] * 2.0                         # 2.0: primary attacking output — conversion threat
            + df["xAG"] * 1.5                       # 1.5: chance creation — secondary attacker role
            + df["shot_creating_actions"] * 0.4     # 0.4: volume signal, broad category
            + df["goal_creating_actions"] * 0.8     # 0.8: high-value direct contributions
            - df["errors_leading_to_shot"] * 1.5    # -1.5: penalise costly defensive mistakes
        )
        return tgi.rename("TGI")

    # ══════════════════════════════════════════════════════════════════════════
    # GK-SPECIFIC FEATURES
    # ══════════════════════════════════════════════════════════════════════════
    def gk_shot_stopping(self, df: pd.DataFrame) -> pd.Series:
        """
        GK Shot Stopping metric: PSxG minus Goals Allowed per 90.

        PSxG (Post-Shot Expected Goals) is the expected goal value of shots
        AFTER they have been struck — accounting for placement and speed.
        PSxG - GA > 0 means the keeper is saving shots harder than average.
        David Raya's elite season in 2023-24 is almost entirely visible here.

        A goalkeeper who faces a 0.8 PSxG chance and saves it has contributed
        +0.8 to this metric. A GK who concedes it contributes -0.2.
        Higher is better. A season total of +5 is elite; -5 is relegation-level.
        """
        return df["PSxG_minus_GA"].rename("GK_SHOT_STOPPING")

    def gk_distribution_quality(self, df: pd.DataFrame) -> pd.Series:
        """
        GK Distribution Quality — proxy for ball-playing ability.

        Lower passes_launched_pct = fewer long kicks = more playing from the
        back. Alisson and Ederson launch significantly fewer passes than
        traditional GKs. Combined with shorter avg_pass_length to reward
        short distributors who support possession play.

        Formula: (100 - passes_launched_pct) / 100 + (1 / avg_pass_length) * 10
        Inverted and normalised so higher = better ball-player.
        """
        distribution = (
            (100 - df["passes_launched_pct"]) / 100   # invert: lower launch% = better ball-player
            + (1 / df["avg_pass_length"].replace(0, np.nan)).fillna(0) * 10  # shorter = more accurate control
        )
        return distribution.rename("GK_DISTRIBUTION_QUALITY")

    def gk_sweeper_keeper_index(self, df: pd.DataFrame) -> pd.Series:
        """
        GK Sweeper Keeper Index — proxy for a GK's role as a 'sweeper'.

        Actions outside the 18-yard box per 90 capture the frequency with which
        a GK rushes out to deal with through-balls and cut out crosses early.
        Ederson and Alisson consistently top this metric in PL data.
        Higher is better (more active GK supporting a high defensive line).
        """
        return df["gk_sweeper_actions"].rename("GK_SWEEPER_KEEPER_INDEX")

    # ══════════════════════════════════════════════════════════════════════════
    # MAIN ENTRY POINTS
    # ══════════════════════════════════════════════════════════════════════════
    def engineer_outfield_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all outfield composite features to a position-specific DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Outfield player DataFrame (position in CB, FB, CM, ST).

        Returns
        -------
        pd.DataFrame
            Input DataFrame with PPI, DAQ, CCC, BRS, PII, TGI columns added.
        """
        df = df.copy()
        df["PPI"] = self.possession_progression_index(df)
        df["DAQ"] = self.defensive_action_quality(df)
        df["CCC"] = self.chance_creation_contribution(df)
        df["BRS"] = self.ball_retention_score(df)
        df["PII"] = self.pressing_intensity_index(df)
        df["TGI"] = self.threat_generation_index(df)
        logger.info(
            "Engineered outfield features for %d players (pos=%s)",
            len(df),
            df["position"].iloc[0] if len(df) > 0 else "unknown"
        )
        return df

    def engineer_gk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add GK-specific composite features.

        Parameters
        ----------
        df : pd.DataFrame
            GK-only DataFrame.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with GK_SHOT_STOPPING, GK_DISTRIBUTION_QUALITY,
            GK_SWEEPER_KEEPER_INDEX columns added.
        """
        df = df.copy()
        df["GK_SHOT_STOPPING"] = self.gk_shot_stopping(df)
        df["GK_DISTRIBUTION_QUALITY"] = self.gk_distribution_quality(df)
        df["GK_SWEEPER_KEEPER_INDEX"] = self.gk_sweeper_keeper_index(df)
        logger.info("Engineered GK features for %d goalkeepers", len(df))
        return df

    def run(self, position_dfs: dict) -> dict:
        """
        Apply feature engineering to all position DataFrames.

        Parameters
        ----------
        position_dfs : dict
            Output of DataPreprocessor.separate_by_position().

        Returns
        -------
        dict
            Same structure, each DataFrame enriched with composite features.
        """
        result = {}
        for pos, df in position_dfs.items():
            if pos == "GK":
                result[pos] = self.engineer_gk_features(df)
            else:
                result[pos] = self.engineer_outfield_features(df)
        logger.info("Feature engineering complete for all positions")
        return result
