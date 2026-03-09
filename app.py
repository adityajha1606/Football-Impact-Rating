# These two lines suppress warning messages that don't affect functionality
# but clutter the terminal output
import sys
import warnings
warnings.filterwarnings("ignore")

# Add current directory to Python's search path so it can find the src/ folder
# When Streamlit runs on a server, it might not automatically know where to look
sys.path.insert(0, ".")

# Standard library and data science imports
import streamlit as st      # the web app framework
import pandas as pd         # data tables
import matplotlib           # chart library
matplotlib.use("Agg")       # CRITICAL: use non-interactive backend
                            # Agg renders to image files instead of trying
                            # to open a GUI window (which servers don't have)

# Import all 6 modules from our src/ package
from src.data_generator import FootballDataGenerator
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.impact_scorer import ImpactScorer
from src.clustering import PlayerArchetypeClusterer
from src.visualizer import FootballVisualizer


# ─────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# Must be the FIRST Streamlit command in the file
# page_title = what appears in the browser tab
# page_icon = emoji or image shown in the tab
# layout="wide" = use full browser width instead of narrow centered column
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Football Impact Rating",
    page_icon="⚽",
    layout="wide",
)


# ─────────────────────────────────────────────────────────────
# PIPELINE FUNCTION WITH CACHING
#
# @st.cache_data is a decorator — it wraps the function with
# caching behaviour. The first time this function runs, it
# executes normally and stores the result in memory. Every
# subsequent call returns the stored result instantly without
# re-running the code.
#
# Without this, every time a user clicks a dropdown, the app
# would regenerate 500 players, preprocess, engineer features,
# score, and cluster — taking 10+ seconds per click. With caching,
# the pipeline runs once when the app first loads, then all
# interactions are instant.
# ─────────────────────────────────────────────────────────────
@st.cache_data
def run_pipeline():
    # Step 1: Generate data
    gen = FootballDataGenerator()
    raw = gen.generate(500, seed=42)

    # Step 2: Preprocess
    pre = DataPreprocessor()
    processed = pre.run(raw)
    pos_dfs = pre.separate_by_position(processed)

    # Step 3: Feature engineering
    eng = FeatureEngineer()
    featured = eng.run(pos_dfs)

    # Step 4: Scoring
    scorer = ImpactScorer()
    scored = scorer.score_all_positions(featured)

    # Step 5: Clustering
    clusterer = PlayerArchetypeClusterer()
    clustered = clusterer.run(scored)

    # Combine all positions into one big DataFrame for player card lookups
    # pd.concat stacks DataFrames vertically (one on top of the other)
    # ignore_index=True resets the row numbering
    all_players = pd.concat(list(clustered.values()), ignore_index=True)

    return all_players, scorer, clustered


# This line actually calls the function and unpacks the three return values
# The first time the app loads, it runs the full pipeline
# After that, returns cached results
all_players, scorer, clustered = run_pipeline()


# ─────────────────────────────────────────────────────────────
# PAGE HEADER
# st.title creates an H1 heading on the webpage
# st.markdown renders markdown text (** ** = bold)
# st.divider draws a horizontal rule
# ─────────────────────────────────────────────────────────────
st.title("⚽ Football Impact Rating")
st.markdown(
    "**Beyond goals and assists** — a composite rating system that captures "
    "what players actually contribute: defensive actions, ball progression, "
    "pressing intensity, chance quality, and retention under pressure."
)
st.divider()


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# st.sidebar.anything creates elements in the left panel
# The sidebar stays visible while the user scrolls the main content
# ─────────────────────────────────────────────────────────────
st.sidebar.title("Select a Player")

# st.sidebar.selectbox creates a dropdown menu in the sidebar
# First argument: the label shown above the dropdown
# Second argument: the list of options
# Returns: whatever the user currently has selected
position = st.sidebar.selectbox(
    "Position",
    ["CM", "ST", "CB", "FB", "GK"],
    help="Scores are normalised within position — a CB's score is relative to other CBs only."
)

# Get the DataFrame for just the selected position
pos_df = clustered[position]

# Create a sorted list of player names for the second dropdown
# sorted() alphabetically sorts the list
player_options = sorted(pos_df["player_name"].tolist())

# Second dropdown: player name within the selected position
# This updates every time the position changes (because player_options changes)
player_name = st.sidebar.selectbox("Player", player_options)

st.sidebar.divider()

# Metric explanations in the sidebar
# st.sidebar.markdown renders text in the sidebar
st.sidebar.markdown("### About the Metrics")
st.sidebar.markdown("""
- **PPI** — Possession Progression Index
- **DAQ** — Defensive Action Quality  
- **CCC** — Chance Creation Contribution
- **BRS** — Ball Retention Score
- **PII** — Pressing Intensity Index
- **TGI** — Threat Generation Index *(attackers)*
""")


# ─────────────────────────────────────────────────────────────
# MAIN CONTENT — renders whenever a player is selected
# In Streamlit, all code runs top to bottom every time the user
# interacts with anything. The if statement prevents errors when
# no player is selected yet.
# ─────────────────────────────────────────────────────────────
if player_name:

    # Generate the player card dictionary from impact_scorer.py
    card = scorer.generate_player_card(player_name, all_players)


    # ── ROW 1: KEY METRICS ────────────────────────────────────
    # st.columns(4) creates 4 equal-width columns side by side
    # The returned objects (col1, col2, col3, col4) are column contexts
    # Anything written inside "with col1:" appears in the first column
    col1, col2, col3, col4 = st.columns(4)

    # st.metric creates a styled metric display with a label above and value below
    col1.metric("Impact Score", f"{card['impact_score']} / 100")
    col2.metric("Position Percentile", f"Top {100 - card['percentile_in_position']:.0f}%")
    col3.metric("Archetype", card["comparable_archetype"])
    col4.metric("Club", card["club"])

    st.divider()


    # ── ROW 2: RADAR CHART + COMPONENT BREAKDOWN ─────────────
    # st.columns([1.4, 1]) creates two unequal columns
    # The numbers are relative widths: left column is 1.4x wider than right
    left, right = st.columns([1.4, 1])

    with left:
        st.subheader("Performance Radar")

        # Create a FootballVisualizer instance to access chart methods
        viz = FootballVisualizer()

        # Generate the radar chart (returns a matplotlib figure object)
        fig = viz.radar_chart(player_name, pos_df)

        # st.pyplot renders a matplotlib figure in the webpage
        # use_container_width=True makes it fill the available column width
        st.pyplot(fig, use_container_width=True)

    with right:
        st.subheader("Component Breakdown")

        # card["component_scores"] is a dictionary like:
        # {"PPI": 0.734, "DAQ": 0.612, ...}
        components = card.get("component_scores", {})

        for comp, val in components.items():
            # .items() iterates over key-value pairs
            # comp = metric name (e.g. "PPI"), val = raw value (e.g. 0.734)

            st.markdown(f"**{comp}**")
            # f"..." is an f-string — Python's way of embedding variables in text
            # f"**{comp}**" with comp="PPI" produces "**PPI**" which markdown renders bold

            # Calculate how this player compares to the rest of their position
            if comp in pos_df.columns:
                col_min = pos_df[comp].min()
                col_max = pos_df[comp].max()

                if col_max > col_min:
                    # Normalise to 0-1 for the progress bar
                    # Formula: (this player's value - minimum) / (maximum - minimum)
                    normalised = (val - col_min) / (col_max - col_min)
                else:
                    normalised = 0.5

                # st.progress creates a filled progress bar
                # value = how full (0.0 to 1.0)
                # text = label shown on the bar
                st.progress(float(normalised), text=f"{val:.3f}")
                # {val:.3f} formats val to 3 decimal places

        st.divider()

        # Display the player's top strength and weakness from the card
        st.markdown(f"✅ **Top Strength:** {card['top_strength']}")
        st.markdown(f"⚠️ **Biggest Weakness:** {card['biggest_weakness']}")
        st.markdown(f"**Age:** {card['age']}  |  **Minutes:** {card['minutes_played']}")


    st.divider()


    # ── ROW 3: TOP 10 TABLE ───────────────────────────────────
    st.subheader(f"Top 10 {position}s by Impact Score")

    # Select the top 10 players in this position by impact score
    # nlargest(10, "impact_score") returns the 10 rows with highest impact_score
    # [["player_name", "club", "archetype_label", "impact_score"]] selects only those 4 columns
    top10 = pos_df.nlargest(10, "impact_score")[
        ["player_name", "club", "archetype_label", "impact_score"]
    ].reset_index(drop=True)
    # reset_index(drop=True) resets row numbers to start from 0

    top10.index += 1
    # Shifts index to start from 1 instead of 0 (looks better in the table)

    top10.columns = ["Player", "Club", "Archetype", "Score"]
    # Rename columns to cleaner display names

    top10["Score"] = top10["Score"].round(1)
    # Round scores to 1 decimal place

    # st.dataframe renders an interactive sortable table
    st.dataframe(top10, use_container_width=True)


    st.divider()


    # ── ROW 4: ARCHETYPE HEATMAP ──────────────────────────────
    st.subheader(f"{position} Archetype Fingerprints")
    st.markdown(
        "Each row is an archetype. Each column is a composite metric. "
        "Values are mean scores (0–100). This shows the statistical "
        "fingerprint of each playing style."
    )

    # Generate the heatmap (returns a matplotlib figure)
    heatmap_fig = viz.archetype_profile_heatmap(pos_df, position)

    st.pyplot(heatmap_fig, use_container_width=True)