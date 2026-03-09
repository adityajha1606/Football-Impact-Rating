"""
data_generator.py
=================
Generates a realistic synthetic dataset of Premier League players whose statistical
profiles mirror FBref per-90 data. The goal is not random noise but football truth:
a generated Erling Haaland-profile striker should look nothing like a generated
Casemiro-profile midfielder, and both should look nothing like a generated
Ederson-profile goalkeeper.

Each position's parameter distributions are grounded in real FBref 2023-24 PL data
ranges so that downstream analysis (composite metrics, clustering, scoring) produces
results that a football analyst would find credible.
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)


# ─── Real Premier League 2023-24 clubs ───────────────────────────────────────
PL_CLUBS = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
    "Burnley", "Chelsea", "Crystal Palace", "Everton", "Fulham",
    "Liverpool", "Luton Town", "Manchester City", "Manchester United",
    "Newcastle United", "Nottingham Forest", "Sheffield United",
    "Tottenham Hotspur", "West Ham United", "Wolverhampton Wanderers",
]

# ─── Player name pools by nationality ────────────────────────────────────────
FIRST_NAMES_EN = ["James", "Jack", "Harry", "Marcus", "Phil", "Declan", "Jordan", "Conor", "Luke", "Ben"]
FIRST_NAMES_ES = ["Alejandro", "Pablo", "Sergio", "Marcos", "Iker", "Diego", "Carlos", "Raul", "Alvaro", "Javi"]
FIRST_NAMES_FR = ["Kylian", "Antoine", "Ousmane", "Lucas", "Clement", "Theo", "Matteo", "Jonathan", "Rayan", "Adrien"]
FIRST_NAMES_BR = ["Gabriel", "Rodrygo", "Vinicius", "Endrick", "Matheus", "Lucas", "Pedro", "Richarlison", "Fabricio", "Eder"]
FIRST_NAMES_DE = ["Niklas", "Julian", "Kai", "Florian", "Leon", "Leroy", "Thomas", "Serge", "Robin", "Timo"]
FIRST_NAMES_AF = ["Mohammed", "Sadio", "Ismaila", "Cheick", "Amadou", "Youssouf", "Ibrahim", "Boubacar", "Seko", "Wilfried"]
FIRST_NAMES_SC = ["Callum", "Andy", "Scott", "Ryan", "Stuart", "John", "Kieran", "Grant", "Lewis", "Craig"]
FIRST_NAMES_NG = ["Victor", "Alex", "Emmanuel", "Ola", "Chukwuemeka", "Taiwo", "Wilfred", "Caleb", "David", "Samuel"]
FIRST_NAMES_IT = ["Federico", "Lorenzo", "Marco", "Nicolo", "Giacomo", "Davide", "Luca", "Andrea", "Mattia", "Francesco"]
FIRST_NAMES_PT = ["Joao", "Ruben", "Bernardo", "Rafael", "Pedro", "Goncalo", "Vitinha", "Diogo", "Bruno", "Renato"]

LAST_NAMES_EN = ["Smith", "Wilson", "Clarke", "Johnson", "White", "Walker", "Shaw", "Maguire", "Rice", "Saka"]
LAST_NAMES_ES = ["Garcia", "Martinez", "Lopez", "Sanchez", "Gomez", "Fernandez", "Torres", "Ramirez", "Flores", "Molina"]
LAST_NAMES_FR = ["Mbappe", "Griezmann", "Dembele", "Hernandez", "Pavard", "Camavinga", "Tchouameni", "Nkunku", "Coman", "Giroud"]
LAST_NAMES_BR = ["Silva", "Santos", "Costa", "Souza", "Oliveira", "Ferreira", "Pereira", "Rodrigues", "Alves", "Neto"]
LAST_NAMES_DE = ["Muller", "Brandt", "Havertz", "Gnabry", "Sule", "Rudiger", "Koch", "Goretzka", "Kimmich", "Hofmann"]
LAST_NAMES_AF = ["Mane", "Diallo", "Kone", "Traore", "Coulibaly", "Camara", "Doumbia", "Diop", "Sow", "Keita"]
LAST_NAMES_SC = ["Robertson", "McTominay", "Tierney", "Armstrong", "McGinn", "Cooper", "Gilmour", "Adams", "Christie", "Dykes"]
LAST_NAMES_NG = ["Osimhen", "Iwobi", "Aribo", "Awoniyi", "Eze", "Lookman", "Balogun", "Elanga", "Chukwueze", "Ndiaye"]
LAST_NAMES_IT = ["Chiesa", "Barella", "Tonali", "Zaniolo", "Locatelli", "Pellegrini", "Immobile", "Verratti", "Raspadori", "Frattesi"]
LAST_NAMES_PT = ["Cancelo", "Dias", "Neves", "Fernandes", "Jota", "Guedes", "Felix", "Horta", "Trincao", "Danilo"]

ALL_FIRST = (FIRST_NAMES_EN + FIRST_NAMES_ES + FIRST_NAMES_FR + FIRST_NAMES_BR +
             FIRST_NAMES_DE + FIRST_NAMES_AF + FIRST_NAMES_SC + FIRST_NAMES_NG +
             FIRST_NAMES_IT + FIRST_NAMES_PT)

ALL_LAST = (LAST_NAMES_EN + LAST_NAMES_ES + LAST_NAMES_FR + LAST_NAMES_BR +
            LAST_NAMES_DE + LAST_NAMES_AF + LAST_NAMES_SC + LAST_NAMES_NG +
            LAST_NAMES_IT + LAST_NAMES_PT)


class FootballDataGenerator:
    """
    Generates a synthetic but statistically realistic FBref-style per-90 dataset
    of Premier League players.

    The distributions for each position are calibrated against real 2023-24 PL
    data. The generator produces data where position identity is the primary driver
    of a player's statistical profile — exactly as in real football analytics.

    Key design decisions:
    - GKs have completely separate stat columns (no xG, no progressive carries)
    - Outfield players have NaN for all GK-specific columns
    - 15 designated outlier players mirror real outlier cases (Haaland xG,
      TAA progressive passes, Casemiro aerials, etc.)
    - Young players (17-21) receive a slight completion-rate penalty
    - Minutes played is correlated with player quality (starters play more)
    """

    # ── Per-position stat distribution parameters ──────────────────────────
    # Format: (mean, std) for each stat. Clipping applied post-sampling.
    POSITION_PARAMS: dict = {
        "ST": {
            "goals": (0.38, 0.18),
            "assists": (0.12, 0.08),
            "xG": (0.35, 0.14),          # Haaland was ~0.72 in 22-23
            "xAG": (0.08, 0.06),
            "progressive_carries": (2.1, 0.9),
            "progressive_passes": (2.8, 1.1),
            "progressive_passes_received": (4.2, 1.5),
            "pressures": (12.0, 4.0),     # strikers press less in low blocks
            "pressure_success_rate": (28.0, 8.0),
            "tackles_won": (0.6, 0.4),
            "interceptions": (0.4, 0.3),
            "aerial_duels_won_pct": (48.0, 14.0),
            "aerials_attempted": (3.2, 1.4),
            "ground_duels_won_pct": (44.0, 10.0),
            "pass_completion_pct": (72.0, 9.0),
            "key_passes": (0.8, 0.5),
            "passes_into_final_third": (0.6, 0.4),
            "shot_creating_actions": (2.4, 1.0),
            "goal_creating_actions": (0.45, 0.25),
            "blocks": (0.3, 0.2),
            "clearances": (0.4, 0.3),
            "errors_leading_to_shot": (0.06, 0.05),
            "carries_into_final_third": (1.8, 0.9),
            "miscontrols": (1.4, 0.7),
            "dispossessed": (1.1, 0.6),
        },
        "CM": {
            "goals": (0.10, 0.08),
            "assists": (0.15, 0.09),
            "xG": (0.09, 0.07),
            "xAG": (0.14, 0.08),
            "progressive_carries": (2.6, 1.0),
            "progressive_passes": (6.8, 2.2),   # highest of any position
            "progressive_passes_received": (3.5, 1.2),
            "pressures": (22.0, 6.0),
            "pressure_success_rate": (32.0, 9.0),
            "tackles_won": (1.4, 0.7),
            "interceptions": (1.0, 0.5),
            "aerial_duels_won_pct": (42.0, 14.0),
            "aerials_attempted": (1.8, 0.9),
            "ground_duels_won_pct": (50.0, 10.0),
            "pass_completion_pct": (82.0, 7.0),
            "key_passes": (1.2, 0.6),
            "passes_into_final_third": (3.2, 1.4),
            "shot_creating_actions": (2.0, 0.9),
            "goal_creating_actions": (0.30, 0.18),
            "blocks": (0.5, 0.3),
            "clearances": (0.6, 0.4),
            "errors_leading_to_shot": (0.04, 0.04),
            "carries_into_final_third": (1.5, 0.8),
            "miscontrols": (1.1, 0.5),
            "dispossessed": (0.9, 0.4),
        },
        "CB": {
            "goals": (0.04, 0.04),
            "assists": (0.04, 0.03),
            "xG": (0.03, 0.03),
            "xAG": (0.03, 0.03),
            "progressive_carries": (1.4, 0.7),
            "progressive_passes": (4.8, 1.8),
            "progressive_passes_received": (1.6, 0.8),
            "pressures": (10.0, 3.5),
            "pressure_success_rate": (30.0, 9.0),
            "tackles_won": (1.2, 0.6),
            "interceptions": (0.9, 0.5),
            "aerial_duels_won_pct": (58.0, 13.0),  # CBs dominate aerials
            "aerials_attempted": (4.8, 1.8),
            "ground_duels_won_pct": (55.0, 10.0),
            "pass_completion_pct": (85.0, 6.5),
            "key_passes": (0.4, 0.3),
            "passes_into_final_third": (2.2, 1.0),
            "shot_creating_actions": (0.6, 0.4),
            "goal_creating_actions": (0.08, 0.06),
            "blocks": (0.9, 0.4),
            "clearances": (4.2, 1.8),             # CBs clear most
            "errors_leading_to_shot": (0.05, 0.04),
            "carries_into_final_third": (0.5, 0.4),
            "miscontrols": (0.8, 0.4),
            "dispossessed": (0.5, 0.3),
        },
        "FB": {
            "goals": (0.07, 0.06),
            "assists": (0.18, 0.10),
            "xG": (0.05, 0.04),
            "xAG": (0.14, 0.07),
            "progressive_carries": (3.1, 1.1),   # highest prog carries
            "progressive_passes": (5.5, 1.8),
            "progressive_passes_received": (3.0, 1.2),
            "pressures": (18.0, 5.0),
            "pressure_success_rate": (31.0, 9.0),
            "tackles_won": (1.5, 0.7),
            "interceptions": (1.1, 0.5),
            "aerial_duels_won_pct": (44.0, 13.0),
            "aerials_attempted": (2.2, 1.0),
            "ground_duels_won_pct": (52.0, 10.0),
            "pass_completion_pct": (80.0, 7.0),
            "key_passes": (1.0, 0.5),
            "passes_into_final_third": (3.0, 1.3),
            "shot_creating_actions": (1.8, 0.8),
            "goal_creating_actions": (0.22, 0.14),
            "blocks": (0.7, 0.4),
            "clearances": (2.4, 1.1),
            "errors_leading_to_shot": (0.04, 0.04),
            "carries_into_final_third": (2.0, 1.0),
            "miscontrols": (1.0, 0.5),
            "dispossessed": (0.8, 0.4),
        },
        "GK": {
            # GK-specific only
            "PSxG_minus_GA": (0.02, 0.18),        # positive = above expected saves
            "save_pct": (72.0, 7.0),
            "passes_launched_pct": (28.0, 12.0),  # lower = more ball-player
            "avg_pass_length": (32.0, 8.0),
            "gk_sweeper_actions": (1.4, 0.8),     # actions outside box per 90
        },
    }

    # ── Outlier injection specifications ──────────────────────────────────
    # Each tuple: (position, stat_to_spike, multiplier, label_hint)
    OUTLIER_SPECS: list = [
        ("ST",  "xG",                    2.8, "Haaland-profile"),
        ("ST",  "goals",                 2.6, "Clinical finisher"),
        ("ST",  "aerial_duels_won_pct",  1.6, "Target man"),
        ("ST",  "pressures",             2.2, "High-press striker"),
        ("FB",  "progressive_carries",   2.4, "Trent-profile"),
        ("FB",  "key_passes",            2.5, "Inverted FB playmaker"),
        ("FB",  "tackles_won",           2.0, "Defensive FB"),
        ("CM",  "progressive_passes",    2.3, "Regista"),
        ("CM",  "pressures",             2.2, "Pressing engine"),
        ("CM",  "goals",                 2.8, "Goal-scoring CM"),
        ("CB",  "clearances",            2.2, "Old-school CB"),
        ("CB",  "progressive_carries",   2.5, "Libero CB"),
        ("CB",  "aerial_duels_won_pct",  1.4, "Aerial colossus"),
        ("GK",  "PSxG_minus_GA",         3.5, "Elite shot-stopper"),
        ("GK",  "gk_sweeper_actions",    3.0, "Sweeper keeper"),
    ]

    def __init__(self) -> None:
        self.rng = None

    def _make_name(self) -> str:
        """Generate a plausible multinational player name."""
        first = self.rng.choice(ALL_FIRST)
        last = self.rng.choice(ALL_LAST)
        return f"{first} {last}"

    def _sample_outfield_player(self, position: str, age: int) -> dict:
        """
        Sample one outfield player's stats from position-specific distributions.

        Young player penalty: players 17-21 receive a -4pp completion rate
        adjustment, reflecting lower technical consistency in real data.
        """
        params = self.POSITION_PARAMS[position]
        row: dict = {"position": position}

        for stat, (mean, std) in params.items():
            val = float(self.rng.normal(mean, std))
            row[stat] = val

        # Young player completion penalty
        if age <= 21:
            row["pass_completion_pct"] -= self.rng.uniform(2.0, 6.0)

        # Clip to realistic ranges
        row["goals"] = max(0.0, row["goals"])
        row["assists"] = max(0.0, row["assists"])
        row["xG"] = np.clip(row["xG"], 0.0, 1.5)
        row["xAG"] = np.clip(row["xAG"], 0.0, 0.8)
        row["progressive_carries"] = max(0.0, row["progressive_carries"])
        row["progressive_passes"] = max(0.0, row["progressive_passes"])
        row["progressive_passes_received"] = max(0.0, row["progressive_passes_received"])
        row["pressures"] = max(0.0, row["pressures"])
        row["pressure_success_rate"] = np.clip(row["pressure_success_rate"], 0.0, 65.0)
        row["tackles_won"] = max(0.0, row["tackles_won"])
        row["interceptions"] = max(0.0, row["interceptions"])
        row["aerial_duels_won_pct"] = np.clip(row["aerial_duels_won_pct"], 0.0, 95.0)
        row["aerials_attempted"] = max(0.0, row["aerials_attempted"])
        row["ground_duels_won_pct"] = np.clip(row["ground_duels_won_pct"], 0.0, 95.0)
        row["pass_completion_pct"] = np.clip(row["pass_completion_pct"], 45.0, 97.0)
        row["key_passes"] = max(0.0, row["key_passes"])
        row["passes_into_final_third"] = max(0.0, row["passes_into_final_third"])
        row["shot_creating_actions"] = max(0.0, row["shot_creating_actions"])
        row["goal_creating_actions"] = max(0.0, row["goal_creating_actions"])
        row["blocks"] = max(0.0, row["blocks"])
        row["clearances"] = max(0.0, row["clearances"])
        row["errors_leading_to_shot"] = np.clip(row["errors_leading_to_shot"], 0.0, 0.4)
        row["carries_into_final_third"] = max(0.0, row["carries_into_final_third"])
        row["miscontrols"] = max(0.0, row["miscontrols"])
        row["dispossessed"] = max(0.0, row["dispossessed"])

        # GK columns are NaN for outfield players
        for gk_col in ["PSxG_minus_GA", "save_pct", "passes_launched_pct",
                        "avg_pass_length", "gk_sweeper_actions"]:
            row[gk_col] = float("nan")

        return row

    def _sample_gk(self) -> dict:
        """Sample a goalkeeper's stats. Outfield columns are set to NaN."""
        params = self.POSITION_PARAMS["GK"]
        row: dict = {"position": "GK"}

        for stat, (mean, std) in params.items():
            val = float(self.rng.normal(mean, std))
            row[stat] = val

        row["save_pct"] = np.clip(row["save_pct"], 50.0, 92.0)
        row["passes_launched_pct"] = np.clip(row["passes_launched_pct"], 5.0, 70.0)
        row["avg_pass_length"] = np.clip(row["avg_pass_length"], 18.0, 60.0)
        row["gk_sweeper_actions"] = max(0.0, row["gk_sweeper_actions"])

        # Outfield columns → NaN for GKs
        for col in ["goals", "assists", "xG", "xAG", "progressive_carries",
                    "progressive_passes", "progressive_passes_received",
                    "pressures", "pressure_success_rate", "tackles_won",
                    "interceptions", "aerial_duels_won_pct", "aerials_attempted",
                    "ground_duels_won_pct", "pass_completion_pct", "key_passes",
                    "passes_into_final_third", "shot_creating_actions",
                    "goal_creating_actions", "blocks", "clearances",
                    "errors_leading_to_shot", "carries_into_final_third",
                    "miscontrols", "dispossessed"]:
            row[col] = float("nan")

        return row

    def _assign_minutes(self, quality_proxy: float, age: int) -> int:
        """
        Assign minutes played. Higher quality players play more (positive
        correlation between statistical quality and selection frequency).
        Older players (>34) may have slightly fewer minutes due to squad rotation.

        quality_proxy is the mean of the player's non-NaN stats after normalisation
        to a rough 0-1 range. This ensures that statistically strong players
        cluster above the 900-minute threshold.
        """
        base = 1500
        quality_bonus = quality_proxy * 1200  # up to ~2700 at peak
        age_penalty = max(0, (age - 33) * 40)
        noise = float(self.rng.normal(0, 200))
        mins = int(base + quality_bonus - age_penalty + noise)
        return int(np.clip(mins, 120, 3420))  # 3420 = 38 games * 90

    def _assign_age(self, position: str) -> int:
        """
        Assign a realistic age per position.
        Strikers peak younger (23-28), CBs peak later (26-31).
        """
        age_params = {
            "GK": (28.0, 4.5),
            "CB": (27.5, 4.0),
            "FB": (25.5, 3.8),
            "CM": (26.5, 4.2),
            "ST": (25.0, 3.5),
        }
        mean, std = age_params[position]
        age = int(self.rng.normal(mean, std))
        return int(np.clip(age, 17, 37))

    def generate(self, n_players: int = 500, seed: int = 42) -> pd.DataFrame:
        """
        Generate a synthetic FBref-style per-90 dataset.

        Parameters
        ----------
        n_players : int
            Total players to generate. Distributed across positions:
            GK=60, CB=110, FB=110, CM=130, ST=90 (reflects squad structure).
        seed : int
            Random seed for full reproducibility.

        Returns
        -------
        pd.DataFrame
            One row per player with all raw per-90 stats, position, club, age,
            minutes_played, and player_name.
        """
        self.rng = np.random.default_rng(seed)
        logger.info("Generating synthetic player dataset (n=%d, seed=%d)", n_players, seed)

        # Position counts that sum to n_players (scaled from defaults)
        scale = n_players / 500
        pos_counts = {
            "GK": int(60 * scale),
            "CB": int(110 * scale),
            "FB": int(110 * scale),
            "CM": int(130 * scale),
            "ST": int(90 * scale),
        }
        # Ensure exact total by adjusting CM
        diff = n_players - sum(pos_counts.values())
        pos_counts["CM"] += diff

        rows = []
        used_names: set = set()

        for position, count in pos_counts.items():
            for _ in range(count):
                # Unique name
                name = self._make_name()
                attempts = 0
                while name in used_names and attempts < 50:
                    name = self._make_name()
                    attempts += 1
                used_names.add(name)

                age = self._assign_age(position)
                club = str(self.rng.choice(PL_CLUBS))

                if position == "GK":
                    row = self._sample_gk()
                else:
                    row = self._sample_outfield_player(position, age)

                # Quality proxy for minutes assignment (rough average of key stats)
                numeric_vals = [v for v in row.values() if isinstance(v, float) and not np.isnan(v)]
                quality_proxy = float(np.mean(numeric_vals)) / 20.0  # normalise roughly to 0-1

                row["player_name"] = name
                row["club"] = club
                row["age"] = age
                row["minutes_played"] = self._assign_minutes(quality_proxy, age)

                rows.append(row)

        df = pd.DataFrame(rows)

        # ── Inject outliers ────────────────────────────────────────────────
        logger.info("Injecting %d outlier players", len(self.OUTLIER_SPECS))
        outlier_rows = []
        for pos, stat, multiplier, label in self.OUTLIER_SPECS:
            age = self._assign_age(pos)
            if pos == "GK":
                row = self._sample_gk()
            else:
                row = self._sample_outfield_player(pos, age)

            # Spike the target stat
            base_mean = self.POSITION_PARAMS[pos].get(stat, (1.0, 0.1))[0]
            row[stat] = float(np.clip(base_mean * multiplier, 0.0, 200.0))

            row["player_name"] = f"Outlier {label} {pos}"
            row["club"] = str(self.rng.choice(PL_CLUBS))
            row["age"] = age
            row["minutes_played"] = int(self.rng.integers(1800, 3200))
            outlier_rows.append(row)

        outlier_df = pd.DataFrame(outlier_rows)
        df = pd.concat([df, outlier_df], ignore_index=True)

        # ── Column ordering ────────────────────────────────────────────────
        meta_cols = ["player_name", "club", "position", "age", "minutes_played"]
        stat_cols = [c for c in df.columns if c not in meta_cols]
        df = df[meta_cols + sorted(stat_cols)]

        logger.info("Dataset generated: %d players, %d columns", len(df), len(df.columns))
        return df
