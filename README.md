# Football Impact Rating ⚽

[![Live App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://football-impact-rating-fqhanzackiv4m4kwguioug.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-deployed-FF4B4B?logo=streamlit&logoColor=white)](https://football-impact-rating-fqhanzackiv4m4kwguioug.streamlit.app/)

> **[Play with the live app here](https://football-impact-rating-fqhanzackiv4m4kwguioug.streamlit.app/)** — no Python, no setup, just open and explore.

---

## So, what is this?

You know that feeling when a midfielder runs 11km, wins every second ball, and presses relentlessly for 90 minutes — and the match report says he had "a quiet game" because he didn't score or assist?

Yeah. This project was built for that.

Football Impact Rating is a player scoring system (0 to 100) that looks at what players *actually do* on the pitch — not just the moments that end up in the scoresheet. It covers Premier League-style data across all five positions, groups players into real tactical archetypes, and wraps everything up in an interactive web app you can use right now.

Under the hood it is a full ML pipeline: data generation, preprocessing, feature engineering, position-aware scoring, K-Means clustering, and six different charts — all deployed live via Streamlit.

---

## The problem with goals and assists (a short rant)

Goals and assists are **terminal events**. They are the last action in a long chain of decisions, movements, duels, and recoveries that actually shaped the game. The problem is they get all the credit.

Consider this: a centre-back who makes 4.2 clearances, wins 1.2 tackles, takes 1.0 interceptions, and dominates 58% of aerial duels across 34 games finishes the season with zero goals and zero assists. He looks invisible in every traditional stat table. He is not invisible. He is the reason his team has a clean sheet record.

This system is built to find that player.

Here is what it picks up that box scores completely miss:

- A CB who breaks the press by carrying the ball 40 metres (that is pure van Dijk)
- A midfielder who wins the ball back in dangerous areas more than anyone else in the league (Kante, obviously)
- A full-back creating more expected assisted goals than most number tens (you know who)
- A striker generating 0.4 xG per game but leaking 0.25 xG through their own errors — their net contribution on those actions is 0.15, not 0.4

Goals and assists tell you what happened at the very end. This tells you why.

---

## See it live

**[football-impact-rating-fqhanzackiv4m4kwguioug.streamlit.app](https://football-impact-rating-fqhanzackiv4m4kwguioug.streamlit.app/)**

Pick a position and a player from the sidebar. You will get:

- An **impact score** from 0 to 100, plus where they rank among all players in their position
- A **radar chart** showing their profile across every metric compared to the position average
- A **component breakdown** so you can see exactly what is driving their score up or down
- A **top 10 leaderboard** for that position
- An **archetype heatmap** showing the statistical fingerprint of each player type

---

## The six metrics (and why they are built the way they are)

Every formula weight has a football reason behind it, not just a mathematical one. Here is each metric, what it asks, and the thinking behind it.

---

### 1. Possession Progression Index (PPI)
*Does this player move the ball forward?*

```
PPI = (progressive_carries x 1.0
     + progressive_passes x 0.7
     + progressive_passes_received x 0.4
     + passes_into_final_third x 0.8) / 4
```

Carries get the top weight because when you drive forward with the ball you are personally taking the risk and earning the reward. Receiving a progressive pass gets the lowest weight — your teammate did most of the work there, you just had to be in the right spot. De Bruyne and Declan Rice both score elite here, but for completely different reasons.

---

### 2. Defensive Action Quality (DAQ)
*Does this player actually win the ball back?*

```
DAQ = (tackles_won x 1.2
     + interceptions x 1.0
     + pressures x (pressure_success_rate / 100) x 0.3
     + (aerial_duels_won_pct / 100) x aerials_attempted x 0.8) / 4
```

The most important detail here is how pressing is handled. Press volume is **multiplied** by success rate, not added to it. Burnley under Sean Dyche in 2022-23 pressed more than almost anyone — and won the ball back almost never. Pure volume would make them look like an elite pressing side. Multiplying by success rate correctly exposes them. For aerials, we weight by attempts too, so winning 60% of 8 headers beats winning 80% of 1.

---

### 3. Chance Creation Contribution (CCC)
*Does this player create chances that actually matter?*

```
CCC = (key_passes x 1.0
     + shot_creating_actions x 0.6
     + goal_creating_actions x 1.5
     + xAG x 2.0) / 4
```

xAG — expected assisted goals — gets the double weight because it measures chance **quality**, not just chance **quantity**. Three assists from tap-ins inside a crowded six-yard box scores much lower than two assists from genuine through-ball one-on-ones. Key passes get the modest weight because a 2-yard cutback to a crowded box counts the same as a 40-yard defence-splitting pass. xAG fixes that.

---

### 4. Ball Retention Score (BRS)
*Does this player actually keep the ball, or just recycle it sideways?*

```
BRS = (pass_completion_pct / 100 x 2.0)
    - (miscontrols x 0.3)
    - (dispossessed x 0.4)
    + (carries_into_final_third x 0.5)
```

Raw pass completion on its own just rewards players who pass backwards all game. The penalties for losing the ball force the metric to distinguish safe-but-useless recyclers from genuine ball-players. And the final-third carry bonus rewards players who take productive risks rather than always taking the safe option. Busquets scores near-perfect here. A stereotypical Dyche-era Burnley midfielder does not.

---

### 5. Pressing Intensity Index (PII)
*Does this player press with a plan, or just chase shadows?*

```
PII = pressures x (pressure_success_rate / 100) x (1 + carries_into_final_third x 0.1)
```

The multiplicative structure is the whole point. Pressing 40 times at 15% gives you 6.0 effective recoveries. Pressing 20 times at 30% also gives you 6.0. Volume alone tells you nothing. The carry bonus on top rewards players who win the ball high up the pitch and immediately go forward — the press-and-go archetype that Jurgen Klopp built an entire identity around.

---

### 6. Threat Generation Index (TGI)
*For attackers: how much danger do you create, minus the danger you give away?*

```
TGI = xG x 2.0
    + xAG x 1.5
    + shot_creating_actions x 0.4
    + goal_creating_actions x 0.8
    - errors_leading_to_shot x 1.5
```

That minus at the end is doing real work. A striker generating 0.4 xG per game but gifting 0.25 xG to the opposition through their own mistake is not contributing 0.4 — they are contributing 0.15 on those actions. The error penalty is weighted above xAG because giving the ball away in a dangerous position has a cost you cannot undo.

---

## How each position is scored

Scores are normalised **within each position group**, not across the whole dataset. A centre-back's DAQ score of 75 means they are in the top 25% of centre-backs on defending. It does not mean they are in the top 25% of all 500 players — that comparison would be meaningless because strikers do a fraction of the defensive work CBs do.

| Position | Top weight | Second weight | The idea behind it |
|----------|-----------|---------------|-------------------|
| CB | DAQ 35% | BRS 25% | Defend first. Ball-playing CBs earn the premium. |
| FB | PPI 30% | CCC 25% | The modern fullback is an attack player who also defends. |
| CM | PPI 30% | BRS 25% | Move it forward, do not give it away. |
| ST | TGI 45% | PPI 20% | Score, create, and press. In that order. |
| GK | Shot Stopping 55% | Distribution 25% | Stop the ball. Then start the play. |

These weights are where the football opinion lives in the code. Change them and you change the whole philosophy of what you think each position is for.

---

## Player archetypes

K-Means clusters each position into tactical archetypes. The labels come from inspecting which metric each cluster scores highest on and matching it to real football language — not from hardcoding cluster numbers.

### Central Midfielders
| Archetype | What the numbers look like | Think of |
|-----------|---------------------------|---------|
| Deep-Lying Playmaker | High BRS and PPI, low pressing | Busquets, Fabinho, Casemiro |
| Box-to-Box Warrior | High PII and DAQ | Kante, Henderson |
| Advanced Playmaker | High CCC, elite xAG | De Bruyne-lite, Eriksen |
| Progressive Carrier | Highest PPI through carries | Declan Rice |

### Centre-Backs
| Archetype | What the numbers look like | Think of |
|-----------|---------------------------|---------|
| Ball-Playing Libero | High BRS and PPI | Laporte, Stones, van Dijk |
| Aerial Enforcer | High DAQ through aerials | Schar, Lindelof |
| Pressing CB | High PII and DAQ | Arsenal high-line profile |
| Complete Defender | Balanced DAQ and BRS | Ruben Dias |

### Full-Backs
| Archetype | What the numbers look like |
|-----------|---------------------------|
| Attacking Wingback | High PPI and CCC — the Trent/Robertson profile |
| Defensive FB | High DAQ — old school, reliable |
| Inverted FB | High CCC — the Cancelo experiment |

### Strikers
| Archetype | What the numbers look like |
|-----------|---------------------------|
| Clinical Finisher | Elite TGI through xG — think Haaland |
| Complete Forward | High TGI and PPI — Firmino, Benzema |
| Pressing Striker | High PII — the side's first line of defence |
| Target Man | High TGI through aerials and hold-up play |

---

## The design decisions worth knowing

These are the questions you will get asked if you put this on your CV, so here are the honest answers.

**Why K-Means and not DBSCAN or hierarchical clustering?**

Player data per position is roughly Gaussian in shape — K-Means handles that well. More importantly, DBSCAN marks unusual players as noise and discards them. That is exactly wrong for scouting. A CB who carries the ball like a midfielder is not an outlier to be discarded — he is the most interesting player in the dataset. We want to find him and cluster him, not throw him away.

**Why MinMaxScaler for scoring but StandardScaler for clustering?**

Scoring needs a number between 0 and 100 that a human can immediately understand. Clustering needs true distances between players in feature space — and for that you need variance preserved. A stat that ranges from 0.1 to 50 should pull the cluster geometry harder than one that ranges from 0.1 to 0.8. MinMax flattens that difference. StandardScaler keeps it.

**Why not just use a neural network?**

Because when a scout shows a sporting director a rating of 73/100 and gets asked why, the answer cannot be "the network said so." The weighted composite can say: high DAQ, weak BRS — a dominant defender who loses the ball too often under pressure. That is a conversation. A neural network is a wall.

**How would you validate this with real data?**

Three ways, all used in the literature:
1. Correlate impact scores with end-of-season Transfermarkt market value changes
2. Compare against manager selection — do high-scoring players play more minutes the following season?
3. Team xG differential with vs without the player on the pitch — the approach behind VAEP (Decroos et al., 2019)

---

## What building this taught me

**The football knowledge is harder than the code.** Deciding that `pressure_success_rate` should multiply `pressures` rather than add to it is not a Python decision. It is a decision about how football actually works. Getting that wrong produces numbers that look fine but mean nothing.

**Normalisation is not just a preprocessing step.** Choosing MinMax vs StandardScaler changes whether a Haaland-profile outlier correctly anchors the cluster geometry or gets squeezed in with everyone else. Every choice in the pipeline has a downstream consequence you have to think through.

**In sports, the outlier is the point.** The standard data science instinct is to remove outliers or treat them as noise. In football analytics that instinct would delete the most valuable player in the dataset. Winsorisation caps their influence on the scale without erasing them from the rankings — they still sit at the top, they just do not pull the floor down for everyone else.

---

## Run it yourself

```bash
# Clone the repo
git clone https://github.com/adityajha1606/football-impact-rating.git
cd football-impact-rating

# Set up a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline — generates data, scores, clusters, saves charts
python main.py

# Or launch the web app locally
streamlit run app.py
```

---

## How the project is organised

```
football-impact-rating/
├── app.py                           # Streamlit web app
├── main.py                          # Full pipeline, runs top to bottom
├── requirements.txt
│
├── src/
│   ├── data_generator.py            # Creates 500 realistic synthetic players
│   ├── preprocessing.py             # Minutes filter, winsorisation, validation
│   ├── feature_engineering.py       # Builds PPI, DAQ, CCC, BRS, PII, TGI
│   ├── impact_scorer.py             # Weighted 0-100 scoring + player cards
│   ├── clustering.py                # K-Means archetype detection
│   └── visualizer.py                # All six chart types, dark theme
│
├── data/
│   ├── raw/players_raw.csv          # Created on first run
│   └── processed/players_processed.csv
│
├── notebooks/
│   └── analysis.ipynb               # Full walkthrough written for a football analytics audience
│
└── outputs/                         # Six PNG charts saved here after running
```

---

## What it is built with

| Tool | What it does here |
|------|------------------|
| Python 3.10+ | Everything |
| pandas | Handles all the player data tables |
| NumPy | Random distributions, array maths, clipping |
| scikit-learn | Scaling, K-Means, silhouette scoring |
| matplotlib | Radar charts, scatter plots, heatmaps, network graphs |
| Streamlit | Turns the whole thing into a live web app |

---

<p align="center">
  Built by <a href="https://github.com/adityajha1606">Aditya Jha</a> &nbsp;·&nbsp;
  <a href="https://football-impact-rating-fqhanzackiv4m4kwguioug.streamlit.app/">Live App</a> &nbsp;·&nbsp;
  Synthetic data, real football logic
</p>
