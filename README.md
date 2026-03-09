# Football Impact Rating 🏴󠁧󠁢󠁥󠁧󠁬󠁥

[![Live App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://football-impact-rating-fqhanzackiv4m4kwguioug.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-deployed-FF4B4B?logo=streamlit&logoColor=white)](https://football-impact-rating-fqhanzackiv4m4kwguioug.streamlit.app/)

> **[▶ Open the live app](https://football-impact-rating-fqhanzackiv4m4kwguioug.streamlit.app/)** — no installation required, runs in your browser.

A player impact rating system for Premier League data that goes beyond goals and assists. Built as a portfolio project demonstrating end-to-end ML engineering: synthetic data generation, preprocessing, composite feature engineering, position-aware scoring, unsupervised clustering, and interactive deployment.

---

## The Problem with Goals + Assists

Traditional football statistics only count **terminal events** — the final action in a long chain of decisions and movements that actually determined the outcome. A striker averaging 0.38 goals per 90 has registered that number from 34 separate performances. Meanwhile a centre-back who made 4.2 clearances, 1.2 tackle wins, 1.0 interceptions, and won 58% of aerial duels in every one of those 34 games finishes the season with **0 goals, 0 assists** — and looks invisible.

Goals and assists measure the outcome of football. This system measures the process.

**What this captures that box scores miss:**
- A centre-back who breaks high lines with 40m progressive carries (van Dijk profile)
- A midfielder who wins the ball back in dangerous areas at elite rate (Kanté profile)
- A full-back who creates more xAG than most attacking midfielders (Trent Alexander-Arnold profile)
- A striker who generates 0.4 xG but gives away a 0.25 xG counter-attack through his own error — net contribution: 0.15

---

## Live Demo

**[https://football-impact-rating-fqhanzackiv4m4kwguioug.streamlit.app/](https://football-impact-rating-fqhanzackiv4m4kwguioug.streamlit.app/)**

Select a position and player from the sidebar to see:
- **Impact score** (0–100) with position percentile
- **Performance radar** across all composite metrics vs position average
- **Component breakdown** showing relative strength per metric
- **Top 10 leaderboard** for that position
- **Archetype fingerprint heatmap** showing the statistical signature of each player type

---

## How It Works — The Six Composite Metrics

All metrics are per-90 minutes. Every formula weight has a football reason, not just a mathematical one.

### 1. Possession Progression Index (PPI)
*"Does this player advance the ball up the pitch?"*

```
PPI = (progressive_carries × 1.0
     + progressive_passes × 0.7
     + progressive_passes_received × 0.4
     + passes_into_final_third × 0.8) / 4
```

Carries get the highest weight because they represent personal risk and personal reward. Receiving a progressive pass gets the lowest weight because the teammate who played it deserves most of the credit. Kevin De Bruyne and Declan Rice both score elite PPI — but via different routes.

---

### 2. Defensive Action Quality (DAQ)
*"Does this player win the ball back effectively?"*

```
DAQ = (tackles_won × 1.2
     + interceptions × 1.0
     + pressures × (pressure_success_rate / 100) × 0.3
     + (aerial_duels_won_pct / 100) × aerials_attempted × 0.8) / 4
```

The critical detail: pressing volume is **multiplied** by success rate, not added to it. Burnley under Dyche in 2022–23 pressed constantly but rarely won the ball back — pure volume would make them look elite. Multiplying by success rate correctly identifies them as low-value pressers. Aerial contribution weights attempt volume so a player who wins 60% of 8 aerials is valued above one who wins 80% of 1.

---

### 3. Chance Creation Contribution (CCC)
*"Does this player create dangerous chances?"*

```
CCC = (key_passes × 1.0
     + shot_creating_actions × 0.6
     + goal_creating_actions × 1.5
     + xAG × 2.0) / 4
```

xAG (expected assisted goals) gets the highest weight because it captures chance **quality** not just quantity. A player who lays off three tap-ins will score lower than one who threads two genuine one-on-ones. Key passes get a modest weight because they treat a 2-yard cutback to a crowded box equally to a 40-yard through-ball.

---

### 4. Ball Retention Score (BRS)
*"Does this player keep the ball under pressure?"*

```
BRS = (pass_completion_pct / 100 × 2.0)
    − (miscontrols × 0.3)
    − (dispossessed × 0.4)
    + (carries_into_final_third × 0.5)
```

Raw completion rate rewards sideways passing. The penalties for miscontrols and dispossession force the metric to distinguish safe-but-useless recyclers from genuine ball-players. The final-third carry bonus rewards players who accept productive risk rather than retreating.

---

### 5. Pressing Intensity Index (PII)
*"Does this player press with purpose and follow through?"*

```
PII = pressures × (pressure_success_rate / 100) × (1 + carries_into_final_third × 0.1)
```

Multiplicative structure is intentional. Pressing 40 times at 15% success rate (6.0 effective recoveries) equals pressing 20 times at 30% (also 6.0). The progressive carry modifier rewards "press-and-go" players who immediately drive forward after winning the ball — compounding the territorial gain from the press.

---

### 6. Threat Generation Index (TGI) — attackers
*"Does this player generate and convert goal threat, net of their own mistakes?"*

```
TGI = xG × 2.0
    + xAG × 1.5
    + shot_creating_actions × 0.4
    + goal_creating_actions × 0.8
    − errors_leading_to_shot × 1.5
```

The error penalty is the most analytically important term. A striker generating 0.4 xG but conceding a 0.25 xG counter through their own mistake is net contributing only 0.15 on those actions. The penalty is weighted at 1.5× — higher than xAG — because conceding from your own error is an asymmetric cost.

---

## Position Weight Philosophy

Component scores are normalised **within each position group**, not globally. A CB's DAQ score of 75 means top 25% of CBs on defending — not top 25% of all players. CBs defend ~3× more per 90 than strikers. Normalising globally would compress all strikers to near-zero on DAQ, destroying meaningful variation.

| Position | Primary Weight | Secondary | Philosophy |
|----------|---------------|-----------|------------|
| **CB** | DAQ 35% | BRS 25% | Defending is the job. Ball-playing CBs command premium fees. |
| **FB** | PPI 30% | CCC 25% | Modern fullback defined by progression and chance creation. |
| **CM** | PPI 30% | BRS 25% | The engine. Cannot progress what you give away. |
| **ST** | TGI 45% | PPI 20% | Primary job: generate and convert threat. |
| **GK** | Shot Stopping 55% | Distribution 25% | Save first. Play from the back second. |

These weights are the **football opinion** baked into the system. Changing them changes the analytical philosophy — and they are the first thing a technical interviewer will challenge you on.

---

## Player Archetypes

Players are clustered into archetypes using K-Means on their composite metric profiles. Labels are assigned by inspecting which feature each cluster centroid ranks highest on — not hardcoded.

### Central Midfielders
| Archetype | Statistical Signature | Real Reference |
|-----------|----------------------|----------------|
| Deep-Lying Playmaker | High BRS + PPI, low PII | Busquets, Fabinho, Casemiro |
| Box-to-Box Warrior | High PII + DAQ | Kanté, Henderson |
| Advanced Playmaker | High CCC, elite xAG | De Bruyne-lite, Eriksen |
| Progressive Carrier | Highest PPI via carries | Declan Rice |

### Centre-Backs
| Archetype | Statistical Signature | Real Reference |
|-----------|----------------------|----------------|
| Ball-Playing Libero | High BRS + PPI | Laporte, Stones, van Dijk |
| Aerial Enforcer | High DAQ via aerials | Schär, Lindelof |
| Pressing CB | High PII + DAQ | Arsenal high-line CBs |
| Complete Defender | Balanced DAQ + BRS | Rúben Dias |

### Full-Backs
| Archetype | Statistical Signature |
|-----------|----------------------|
| Attacking Wingback | High PPI + CCC — Trent/Robertson |
| Defensive FB | High DAQ — traditional role |
| Inverted FB | High CCC — Cancelo-style |

### Strikers
| Archetype | Statistical Signature |
|-----------|----------------------|
| Clinical Finisher | Elite TGI via xG — Haaland profile |
| Complete Forward | High TGI + PPI — Firmino/Benzema |
| Pressing Striker | High PII — first line of press |
| Target Man | High TGI via aerials and hold-up |

---

## Technical Decisions

**Why K-Means and not DBSCAN or hierarchical clustering?**

- Player feature space is approximately Gaussian per position. K-Means produces compact spherical clusters that map cleanly to the 4–6 archetypes scouts actually use in conversation. DBSCAN would mark outlier players as noise — analytically wrong, because an outlier player (a CB who carries more than most midfielders) is the *most* interesting scouting target. We want to find them, not discard them.

**Why MinMaxScaler for scoring but StandardScaler for clustering?**

- Scoring needs bounded 0–100 output that humans read immediately. Clustering needs true variance preserved — a feature spanning 0.1–50 should exert more geometric influence than one spanning 0.1–0.8. MinMax equalises these ranges artificially. StandardScaler preserves relative variance so genuine outliers pull the cluster geometry correctly.

**Why not a neural network?**

- Interpretability. A scout needs to explain to a sporting director *why* a player scores 73/100. A neural network cannot say "high DAQ but weak BRS — dominant defender who loses the ball too often." The weighted composite is fully auditable. Every number traces back to a formula with a football reason.

**How would you validate this against real outcomes?**

- Three approaches from the football analytics literature:
1. Correlate impact score with end-of-season Transfermarkt market value change
2. Correlate with manager selection frequency (minutes played the following season)
3. Team xG differential in games with vs without the player — the gold-standard VAEP approach (Decroos et al., 2019)

---

## What I Learned

**Football knowledge is the constraint, not the code.** Deciding that `pressure_success_rate` should be *multiplied* by `pressures` rather than added to them is a football decision. Every formula weight has a reason rooted in how the sport actually works — and defending those reasons is harder than writing the code.

**Normalisation choices have large downstream consequences.** Choosing MinMax vs StandardScaler changes whether a Haaland-profile player pulls the cluster geometry correctly. These are not arbitrary defaults — they reflect what you need "distance" to mean in each layer of the pipeline.

**Outliers are signal, not noise, in sports data.** The instinct to remove outliers is analytically wrong here. A 35-xG season IS the finding. Winsorisation preserves rank-ordering while preventing mathematical dominance of scaled outputs — preserving the outlier's status as the top value without letting it collapse everyone else to the bottom 5% of a 0–100 scale.

---

## Run Locally

```bash
# Clone the repository
git clone https://github.com/adityajha1606/football-impact-rating.git
cd football-impact-rating

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the full data pipeline (generates data, scores, clusters, saves charts)
python main.py

# Launch the interactive web app locally
streamlit run app.py
```

---

## Project Structure

```
football-impact-rating/
├── app.py                           # Streamlit web application
├── main.py                          # Full pipeline runner (CLI)
├── requirements.txt
│
├── src/
│   ├── data_generator.py            # Synthetic FBref-style data (500 players)
│   ├── preprocessing.py             # Minutes filter, winsorisation, validation
│   ├── feature_engineering.py       # PPI, DAQ, CCC, BRS, PII, TGI
│   ├── impact_scorer.py             # Weighted 0–100 scoring engine + player cards
│   ├── clustering.py                # K-Means archetype identification
│   └── visualizer.py                # StatsBomb-aesthetic matplotlib charts
│
├── data/
│   ├── raw/players_raw.csv          # Generated on first run
│   └── processed/players_processed.csv
│
├── notebooks/
│   └── analysis.ipynb               # End-to-end walkthrough with football context
│
└── outputs/                         # PNG charts (generated on run)
    ├── cm_radar_top_scorer.png
    ├── cm_scatter_daq_vs_ppi.png
    ├── cm_impact_distribution.png
    ├── cm_archetype_heatmap.png
    ├── cm_comparison_spider.png
    └── cm_similarity_network.png
```

---

## Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| pandas | Data manipulation and per-90 stats |
| NumPy | Statistical distributions, array operations |
| scikit-learn | MinMaxScaler, StandardScaler, KMeans, silhouette score |
| matplotlib | All visualisations (radar, scatter, heatmap, network) |
| Streamlit | Interactive web app and deployment |

---

<p align="center">
  Built by <a href="https://github.com/adityajha1606">Aditya Jha</a> &nbsp;·&nbsp;
  <a href="https://football-impact-rating-fqhanzackiv4m4kwguioug.streamlit.app/">Live App</a> &nbsp;·&nbsp;
  Data is synthetic but statistically grounded in real FBref Premier League distributions
</p>
