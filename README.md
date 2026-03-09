# Football Impact Rating 🏴󠁧󠁢󠁥󠁧󠁬󠁿

A player impact rating system built for Premier League data that goes beyond goals and assists.

---

## The Problem with Goals + Assists

Traditional football statistics have a fundamental flaw: they only count high-variance, low-frequency terminal events. A striker who averages 0.38 goals/90 over a season has registered that number from roughly 34 separate 90-minute performances. Contrast this with a centre-back who has made 4.2 clearances, 1.2 tackle wins, 1.0 interceptions, and won 58% of aerial duels in every single one of those same 34 games — a player whose contribution to keeping clean sheets is vast, yet who ends the season with 0 goals and 0 assists.

Goals and assists are the outcomes of football, not the process. This system measures the process.

**What this captures that box scores miss:**
- A centre-back who breaks high lines by carrying the ball 40m (Virgil van Dijk profile)
- A midfielder who wins the ball back in dangerous areas at elite frequency (Kanté profile)
- A full-back who creates more expected assisted goals than most attacking midfielders (Trent TAA profile)
- A striker who generates 0.4 xG but concedes 0.25 xG-worth of chances through his own errors (net-negative action)

---

## The Six Composite Metrics

All metrics are per-90 minutes and designed around football conceptual dimensions:

### 1. Possession Progression Index (PPI)
**"Does this player advance the ball up the pitch?"**

Combines progressive carries (highest weight — personal risk, personal reward), progressive passes (slightly discounted — credit shared between passer and receiver), progressive passes received (lowest weight — good positioning but partly teammate-dependent), and passes into the final third.

*Kevin De Bruyne and Declan Rice both score elite PPI, but via different routes.*

### 2. Defensive Action Quality (DAQ)
**"Does this player win the ball back effectively?"**

Tackles won (highest weight — physical ball recovery), interceptions (anticipatory defending), effective pressing (pressures × success rate — volume without success is noise), and aerial duels won (% × attempts — win rate alone ignores a CB who wins 80% of 1 aerial vs 60% of 8).

*N'Golo Kanté would score elite. Sean Dyche's Burnley 2022-23 press volume would score low — they pressed desperately but infrequently won the ball back.*

### 3. Chance Creation Contribution (CCC)
**"Does this player create dangerous chances?"**

Key passes (standard metric, modest weight), shot creating actions (broad volume signal, discounted), goal creating actions (premium weight — directly led to goals), and xAG — expected assisted goals — which gets the highest weight because it captures chance *quality*, not just quantity.

*A player who assists 3 goals from tap-ins will score lower than one who creates 2 goals from genuine through-ball chances. xAG is the forensic evidence.*

### 4. Ball Retention Score (BRS)
**"Does this player keep the ball under pressure?"**

Pass completion rate (but penalised to avoid rewarding sideways passers) minus miscontrols minus dispossessions plus a bonus for carrying into the final third. The final-third carry bonus separates *productive risk-taking* from timid ball-recycling.

*Sergio Busquets has near-perfect BRS. Wilfried Zaha had good BRS despite high dispossession because his carries_into_final_third bonus offset the penalty.*

### 5. Pressing Intensity Index (PII)
**"Does this player press with purpose and follow through?"**

Multiplicative structure: pressures × success rate × (1 + carries_into_final_third bonus). Pressing without winning the ball contributes almost nothing. The carry bonus captures "press-and-go" players who compound territorial gains.

### 6. Threat Generation Index (TGI) — attackers
**"Does this player generate and convert goal threat, net of defensive errors?"**

xG (highest weight — primary attacker job), xAG (slightly lower), shot creating and goal creating actions, **minus errors leading to shot** (penalised at 1.5× — a striker who turns over possession for a counter-attack 0.25 xG chance has reduced their net contribution significantly).

---

## Position Weight Philosophy

All component scores are normalised to 0-100 **within their position group**, then combined with position-specific weights.

**Why within-position?** A centre-back's DAQ of 75 means she's in the top 25% of CBs on defending. A striker's DAQ of 75 means she's in the top 25% of strikers on defending. These are different things — CBs defend ~3× more per 90 than strikers. Cross-position normalisation would compress all strikers to near-zero on DAQ, destroying any variation in that dimension.

| Position | Primary Weight | Philosophy |
|----------|---------------|-----------|
| CB | DAQ (35%) | Defending is the job. Ball-playing CBs valued at BRS (25%). |
| FB | PPI (30%) | Modern fullback is defined by progression. CCC (25%) reflects Trent/Robertson evolution. |
| CM | PPI (30%) | The engine of the team. BRS (25%) because you can't progress what you give away. |
| ST | TGI (45%) | Primary job: score and create. PPI (20%) + PII (20%) for work rate profile. |
| GK | Shot Stopping (55%) | Save what you face first. Distribution (25%) for modern GK role. |

---

## Player Archetypes

### Central Midfielders
| Archetype | Profile | Real Reference |
|-----------|---------|---------------|
| Deep-Lying Playmaker | High BRS + PPI, low pressing | Busquets, Fabinho, Casemiro |
| Box-to-Box Warrior | High PII + DAQ | Kanté, Henderson, Kovacic |
| Advanced Playmaker | High CCC, elite xAG | De Bruyne-lite, Eriksen |
| Progressive Carrier | Highest PPI via carries | Declan Rice, Fernandinho |

### Centre-Backs
| Archetype | Profile | Real Reference |
|-----------|---------|---------------|
| Ball-Playing Libero | High BRS + PPI | Laporte, Stones, van Dijk |
| Aerial Enforcer | High DAQ via aerials/clearances | Schär, Lindelof |
| Pressing CB | High PII + DAQ | Arsenal CBs in high-line system |
| Complete Defender | Balanced DAQ + BRS | Rúben Dias |

### Full-Backs
| Archetype | Profile |
|-----------|---------|
| Attacking Wingback | High PPI + CCC — Trent/Robertson |
| Defensive FB | High DAQ — traditional role |
| Inverted FB | High CCC — Cancelo-style |
| Balanced FB | Balanced across dimensions |

---

## Example Player Cards

*(Generated from synthetic data — names are illustrative)*

### Top Central Midfielder
```
┌────────────────────────────────────────────────────────────┐
│  CENTRAL MIDFIELDER PLAYER CARD                            │
├────────────────────────────────────────────────────────────┤
│  Impact Score: 87.3/100    Percentile: Top 3% of CMs      │
│  Archetype: Progressive Carrier                            │
│  Top Strength: PPI   ↓ Biggest Weakness: DAQ              │
│  Component Scores:                                         │
│    PPI: 0.94  BRS: 0.81  CCC: 0.67  DAQ: 0.38  PII: 0.72 │
└────────────────────────────────────────────────────────────┘
```

### Top Striker
```
┌────────────────────────────────────────────────────────────┐
│  STRIKER PLAYER CARD                                       │
├────────────────────────────────────────────────────────────┤
│  Impact Score: 91.2/100    Percentile: Top 1% of STs      │
│  Archetype: Clinical Finisher                              │
│  Top Strength: TGI   ↓ Biggest Weakness: CCC              │
│  Component Scores:                                         │
│    TGI: 0.97  PPI: 0.62  PII: 0.58  BRS: 0.70  CCC: 0.31 │
└────────────────────────────────────────────────────────────┘
```

---

## Technical Decisions

**Why K-Means for clustering?**
Player feature space is approximately Gaussian per position (by construction). K-Means produces compact, spherical clusters that map cleanly to the 4-6 archetypes scouts actually use. DBSCAN would mark outlier players as noise — wrong analytically. Outlier players (a CB who carries more than most midfielders) are the *most* interesting scouting targets, not discardable observations.

**Why MinMax for scoring, StandardScaler for clustering?**
Scoring needs bounded 0-100 output for human readability. Clustering needs true variance preserved — a feature spanning 0.1 to 50 should exert more geometric influence than one spanning 0.1 to 0.8.

**Why not a neural network?**
Interpretability. A scout needs to explain to a sporting director WHY a player scores 73/100. A neural network cannot say "high DAQ but weak BRS — dominant defender who loses the ball too often under pressure." The weighted composite is fully auditable.

---

## What I Learned

Building this project forced three key analytical skills:

1. **Football knowledge is the constraint, not the code.** Deciding that `pressure_success_rate` should be multiplied by `pressures` rather than added to them is a football decision, not a maths decision. Every formula has a reason rooted in how the sport actually works.

2. **Normalisation choices have massive downstream consequences.** Choosing MinMax vs StandardScaler changes whether Haaland's xG pulls the cluster geometry correctly. These aren't arbitrary — they reflect what you need distance to *mean* in each layer.

3. **Outliers are signal, not noise, in sports data.** The standard instinct to remove outliers is analytically wrong here. A 35-xG season IS the finding. Winsorisation preserves rank while preventing mathematical dominance of scaled outputs.

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full pipeline
python main.py

# 3. View the analysis notebook
jupyter notebook notebooks/analysis.ipynb
```

All outputs (charts) are saved to `outputs/`.

---

## Project Structure

```
football-impact-rating/
├── data/
│   ├── raw/players_raw.csv          # synthetic FBref-style data
│   └── processed/players_processed.csv
├── src/
│   ├── data_generator.py            # realistic synthetic data with position distributions
│   ├── preprocessing.py             # filtering, winsorisation, validation
│   ├── feature_engineering.py       # PPI, DAQ, CCC, BRS, PII, TGI composite metrics
│   ├── impact_scorer.py             # weighted scoring engine + player cards
│   ├── clustering.py                # K-Means archetype identification
│   └── visualizer.py                # StatsBomb-aesthetic charts
├── notebooks/analysis.ipynb
├── outputs/                         # PNG chart outputs
├── main.py
└── requirements.txt
```

---

*Built as a portfolio project demonstrating ML engineering in a football analytics context. Data is synthetic but statistically grounded in real FBref Premier League distributions.*
