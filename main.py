"""
main.py
=======
Full pipeline runner for the football-impact-rating system.

Executes all stages in order:
1. Generate synthetic data
2. Preprocess (filter, winsorise, validate)
3. Feature engineering (composite metrics)
4. Impact scoring (0-100 per position)
5. Archetype clustering
6. Visualisations (CM position demo)
7. Top 5 per position report
8. Player cards
9. Cross-position comparison warning demo
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure src/ is importable when running from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.data_generator import FootballDataGenerator
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.impact_scorer import ImpactScorer
from src.clustering import PlayerArchetypeClusterer
from src.visualizer import FootballVisualizer

# ── Logging configuration ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    print("\n" + "=" * 70)
    print("  FOOTBALL IMPACT RATING — FULL PIPELINE")
    print("=" * 70 + "\n")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 1: DATA GENERATION
    # ──────────────────────────────────────────────────────────────────────────
    print("▶  STAGE 1: Generating synthetic player dataset...")
    generator = FootballDataGenerator()
    raw_df = generator.generate(n_players=500, seed=42)

    Path("data/raw").mkdir(parents=True, exist_ok=True)
    raw_df.to_csv("data/raw/players_raw.csv", index=False)
    print(f"   ✓ Generated {len(raw_df)} players → data/raw/players_raw.csv")
    print(f"   Positions: {raw_df['position'].value_counts().to_dict()}")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 2: PREPROCESSING
    # ──────────────────────────────────────────────────────────────────────────
    print("\n▶  STAGE 2: Preprocessing (filter minutes, winsorise, validate)...")
    preprocessor = DataPreprocessor()
    processed_df = preprocessor.run(raw_df, min_minutes=900)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    processed_df.to_csv("data/processed/players_processed.csv", index=False)
    print(f"   ✓ Processed {len(processed_df)} players → data/processed/players_processed.csv")

    # Separate by position
    position_dfs = preprocessor.separate_by_position(processed_df)
    for pos, df in position_dfs.items():
        print(f"   {pos}: {len(df)} players")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 3: FEATURE ENGINEERING
    # ──────────────────────────────────────────────────────────────────────────
    print("\n▶  STAGE 3: Feature engineering (composite metrics)...")
    engineer = FeatureEngineer()
    featured_dfs = engineer.run(position_dfs)
    print("   ✓ Composite metrics added: PPI, DAQ, CCC, BRS, PII, TGI (outfield)")
    print("   ✓ GK metrics added: GK_SHOT_STOPPING, GK_DISTRIBUTION_QUALITY, GK_SWEEPER_KEEPER_INDEX")

    # Quick sanity check — show CM feature ranges
    cm_features = featured_dfs["CM"][["PPI", "DAQ", "CCC", "BRS", "PII"]].describe()
    print("\n   CM composite feature summary:")
    print(cm_features.loc[["mean", "min", "max"]].round(3).to_string())

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 4: IMPACT SCORING
    # ──────────────────────────────────────────────────────────────────────────
    print("\n▶  STAGE 4: Computing impact scores (0-100 within position)...")
    scorer = ImpactScorer()
    scored_dfs = scorer.score_all_positions(featured_dfs)

    for pos, df in scored_dfs.items():
        top1 = df.nlargest(1, "impact_score").iloc[0]
        print(f"   {pos} top scorer: {top1['player_name']} ({top1['club']}) — {top1['impact_score']:.1f}/100")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 5: CLUSTERING
    # ──────────────────────────────────────────────────────────────────────────
    print("\n▶  STAGE 5: Archetype clustering (K-Means per position)...")
    clusterer = PlayerArchetypeClusterer()
    clustered_dfs = clusterer.run(scored_dfs)

    for pos, df in clustered_dfs.items():
        if "archetype_label" in df.columns:
            counts = df["archetype_label"].value_counts().to_dict()
            print(f"   {pos} archetypes: {counts}")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 6: VISUALISATIONS (CM DEMO)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n▶  STAGE 6: Generating visualisations (CM position demo)...")
    viz = FootballVisualizer()

    cm_df = clustered_dfs["CM"]

    # Select sample player: top scorer in CM
    top_cm = cm_df.nlargest(1, "impact_score").iloc[0]["player_name"]
    print(f"   Sample player for radar/similarity: {top_cm}")

    # Top 4 CMs for comparison spider
    top4_cms = cm_df.nlargest(4, "impact_score")["player_name"].tolist()

    Path("outputs").mkdir(exist_ok=True)

    try:
        viz.radar_chart(top_cm, cm_df, "cm_radar_top_scorer.png")
        print("   ✓ Radar chart")

        viz.position_scatter(cm_df, "CM", "DAQ", "PPI", "cm_scatter_daq_vs_ppi.png")
        print("   ✓ Position scatter (DAQ vs PPI)")

        viz.impact_score_distribution(cm_df, "CM", filename="cm_impact_distribution.png")
        print("   ✓ Impact score distribution")

        viz.archetype_profile_heatmap(cm_df, "CM", "cm_archetype_heatmap.png")
        print("   ✓ Archetype heatmap")

        viz.player_comparison_spider(top4_cms, cm_df, "cm_comparison_spider.png")
        print("   ✓ Comparison spider (top 4 CMs)")

        viz.similarity_network(top_cm, cm_df, filename="cm_similarity_network.png")
        print("   ✓ Similarity network")

        print("   All charts saved to outputs/")
    except Exception as e:
        print(f"   ⚠ Chart generation error: {e}")
        logger.exception("Chart generation failed")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 7: TOP 5 PER POSITION
    # ──────────────────────────────────────────────────────────────────────────
    print("\n▶  STAGE 7: Top 5 players per position\n")
    print("=" * 70)
    for pos, df in clustered_dfs.items():
        print(f"\n  {pos}:")
        print(f"  {'Player':<28} {'Club':<22} {'Archetype':<26} {'Score':>6}")
        print(f"  {'-'*28} {'-'*22} {'-'*26} {'-'*6}")
        top5 = df.nlargest(5, "impact_score")
        for _, row in top5.iterrows():
            arch = row.get("archetype_label", "N/A")
            print(f"  {row['player_name']:<28} {row['club']:<22} {str(arch):<26} {row['impact_score']:>6.1f}")
    print("\n" + "=" * 70)

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 8: PLAYER CARDS
    # ──────────────────────────────────────────────────────────────────────────
    print("\n▶  STAGE 8: Player cards\n")

    # Build a combined dataframe for card lookup
    all_players_df = pd.concat(list(clustered_dfs.values()), ignore_index=True)

    # Best striker
    top_st_name = clustered_dfs["ST"].nlargest(1, "impact_score").iloc[0]["player_name"]
    st_card = scorer.generate_player_card(top_st_name, all_players_df)
    _print_player_card(st_card, "STRIKER")

    # Best CM
    top_cm_name = clustered_dfs["CM"].nlargest(1, "impact_score").iloc[0]["player_name"]
    cm_card = scorer.generate_player_card(top_cm_name, all_players_df)
    _print_player_card(cm_card, "CENTRAL MIDFIELDER")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 9: CROSS-POSITION COMPARISON WARNING DEMO
    # ──────────────────────────────────────────────────────────────────────────
    print("\n▶  STAGE 9: Cross-position comparison warning demonstration")
    print("   (Comparing top ST vs top CM — different position normalisations)\n")

    comparison_players = [top_st_name, top_cm_name]
    result = scorer.compare_players(comparison_players, all_players_df)
    if not result.empty:
        print(result.to_string())

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("  Charts saved to: outputs/")
    print("  Raw data:        data/raw/players_raw.csv")
    print("  Processed data:  data/processed/players_processed.csv")
    print("=" * 70 + "\n")


def _print_player_card(card: dict, position_label: str) -> None:
    """Pretty-print a player card to stdout."""
    if "error" in card:
        print(f"  Error: {card['error']}")
        return

    print(f"\n  ┌{'─'*60}┐")
    print(f"  │  {position_label} PLAYER CARD{' '*(60-len(position_label)-17)}│")
    print(f"  ├{'─'*60}┤")
    print(f"  │  Name:      {card['name']:<46}│")
    print(f"  │  Club:      {card['club']:<46}│")
    print(f"  │  Position:  {card['position']:<46}│")
    print(f"  │  Age:       {card['age']:<46}│")
    print(f"  │  Minutes:   {card['minutes_played']:<46}│")
    print(f"  ├{'─'*60}┤")
    score_str = f"{card['impact_score']}/100"
    print(f"  │  IMPACT SCORE:  {score_str:<43}│")
    pct_str = f"Top {100-card['percentile_in_position']:.0f}% of {card['position']}s"
    print(f"  │  Percentile:    {pct_str:<43}│")
    print(f"  ├{'─'*60}┤")
    print(f"  │  Component Scores:{'':42}│")
    for comp, val in card.get("component_scores", {}).items():
        comp_line = f"    {comp}: {val:.3f}"
        print(f"  │  {comp_line:<58}│")
    print(f"  ├{'─'*60}┤")
    strength_str = f"↑ {card['top_strength']}"
    weakness_str = f"↓ {card['biggest_weakness']}"
    print(f"  │  Top Strength:    {strength_str:<41}│")
    print(f"  │  Biggest Weakness:{weakness_str:<41}│")
    print(f"  │  Archetype:       {str(card['comparable_archetype']):<41}│")
    print(f"  └{'─'*60}┘")


if __name__ == "__main__":
    main()
