"""
clustering.py
=============
Identifies player archetypes — the statistical fingerprints that scout
reports describe verbally (box-to-box, deep-lying playmaker, ball-playing
libero) — through unsupervised clustering of composite performance metrics.

The key insight is that football already has a rich vocabulary of player
archetypes. This module's job is not to DISCOVER new categories but to
ASSIGN players to the categories that scouts, managers, and analysts already
use. Cluster labels are manually assigned by inspecting centroids against
football knowledge — the maths identifies groupings, the football brain names them.

WHY K-MEANS OVER ALTERNATIVES:
- DBSCAN would mark unusual players as 'noise' — analytically wrong. An outlier
  player (a CB who carries more than most midfielders, a GK who plays like a CB)
  is the MOST interesting scouting target. We cannot afford to discard them.
- Hierarchical clustering produces dendrograms that are difficult to map to
  discrete archetypes. Scouts need a clear category, not a tree structure.
- K-Means produces compact, spherical clusters in feature space. Player
  performance data in composite-feature space is approximately Gaussian per
  position (by construction of our data generation), making K-Means optimal.
- Stability: n_init=20 runs ensure the solution is not a local minimum.
  With good data, K-Means converges to the same solution across runs.
- Interpretability: 4-6 clusters per position is exactly the number of
  archetypes a scout would verbally distinguish. More clusters lose meaning
  (scouts don't talk about 8 types of CM); fewer miss real distinctions.

WHY STANDARDSCALER NOT MINMAX FOR CLUSTERING:
Clustering operates in Euclidean distance space. MinMax scaling compresses all
features to [0,1] but destroys variance information — a feature with natural
range 0.1-0.8 looks as spread as one with range 0.1-50 after MinMax. StandardScaler
preserves the relative variance structure, so genuinely extreme values (Haaland
xG, TAA progressive carries) pull the cluster geometry correctly, placing outliers
at the periphery rather than in the middle of artificially equal-range dimensions.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ── Composite features used for clustering (by position) ─────────────────────
CLUSTER_FEATURES: Dict[str, List[str]] = {
    "CB": ["PPI", "DAQ", "CCC", "BRS", "PII"],
    "FB": ["PPI", "CCC", "DAQ", "BRS", "PII"],
    "CM": ["PPI", "DAQ", "CCC", "BRS", "PII"],
    "ST": ["TGI", "PPI", "PII", "BRS", "CCC"],
    "GK": ["GK_SHOT_STOPPING", "GK_DISTRIBUTION_QUALITY", "GK_SWEEPER_KEEPER_INDEX"],
}

# ── Archetype label definitions per position ──────────────────────────────────
# Each archetype is identified by the feature it RANKS HIGHEST on in the centroid.
# This makes label assignment general (not tied to specific cluster numbers).

CM_ARCHETYPES: Dict[str, Dict] = {
    "Deep-Lying Playmaker": {
        "primary_feature": "BRS",
        "secondary_feature": "PPI",
        "description": "High BRS + high PPI, low PII — Busquets/Fabinho/Casemiro profile",
        "low_feature": "PII",
    },
    "Box-to-Box Warrior": {
        "primary_feature": "PII",
        "secondary_feature": "DAQ",
        "description": "High PII + high DAQ — Kanté/Henderson pressing engine",
        "low_feature": "CCC",
    },
    "Advanced Playmaker": {
        "primary_feature": "CCC",
        "secondary_feature": "TGI" if "TGI" in CLUSTER_FEATURES["CM"] else "PPI",
        "description": "High CCC — Eriksen/De Bruyne-lite creative hub",
        "low_feature": "DAQ",
    },
    "Progressive Carrier": {
        "primary_feature": "PPI",
        "secondary_feature": "BRS",
        "description": "Highest PPI via carries — Declan Rice/Fernandinho engine",
        "low_feature": "CCC",
    },
}

CB_ARCHETYPES: Dict[str, Dict] = {
    "Ball-Playing Libero": {
        "primary_feature": "BRS",
        "secondary_feature": "PPI",
        "description": "High BRS + high PPI — Laporte/van Dijk progressive passer",
    },
    "Aerial Enforcer": {
        "primary_feature": "DAQ",
        "secondary_feature": "DAQ",  # DAQ driven by aerials
        "description": "High DAQ via aerials + clearances — Schär/Lindelof profile",
    },
    "Pressing CB": {
        "primary_feature": "PII",
        "secondary_feature": "DAQ",
        "description": "High PII + high DAQ — modern high-line aggressive CB",
    },
    "Complete Defender": {
        "primary_feature": "DAQ",
        "secondary_feature": "BRS",
        "description": "Balanced DAQ + BRS — the all-round defender",
    },
}

FB_ARCHETYPES: Dict[str, Dict] = {
    "Attacking Wingback": {
        "primary_feature": "PPI",
        "secondary_feature": "CCC",
        "description": "High PPI + high CCC — Trent/Robertson offensive FB",
    },
    "Defensive FB": {
        "primary_feature": "DAQ",
        "secondary_feature": "BRS",
        "description": "High DAQ — traditional defensive-minded fullback",
    },
    "Inverted FB": {
        "primary_feature": "CCC",
        "secondary_feature": "BRS",
        "description": "High CCC — Cancelo-style tucking inside to create",
    },
    "Balanced FB": {
        "primary_feature": "BRS",
        "secondary_feature": "PPI",
        "description": "Balanced contributions across all dimensions",
    },
}

ST_ARCHETYPES: Dict[str, Dict] = {
    "Clinical Finisher": {
        "primary_feature": "TGI",
        "secondary_feature": "BRS",
        "description": "Elite TGI — Haaland/Kane goal machine",
    },
    "Complete Forward": {
        "primary_feature": "TGI",
        "secondary_feature": "PPI",
        "description": "High TGI + high PPI — Firmino/Benzema all-round threat",
    },
    "Pressing Striker": {
        "primary_feature": "PII",
        "secondary_feature": "TGI",
        "description": "High PII — tireless pressing striker, disrupts build-up",
    },
    "Target Man": {
        "primary_feature": "TGI",
        "secondary_feature": "CCC",
        "description": "Strong TGI via aerials — hold-up play and combination striker",
    },
}

GK_ARCHETYPES: Dict[str, Dict] = {
    "Elite Shot-Stopper": {
        "primary_feature": "GK_SHOT_STOPPING",
        "secondary_feature": "GK_SWEEPER_KEEPER_INDEX",
        "description": "Top PSxG-GA — best shot-stopper in dataset",
    },
    "Sweeper Keeper": {
        "primary_feature": "GK_SWEEPER_KEEPER_INDEX",
        "secondary_feature": "GK_DISTRIBUTION_QUALITY",
        "description": "High sweeper actions + good distribution — Ederson/Alisson",
    },
    "Ball-Playing GK": {
        "primary_feature": "GK_DISTRIBUTION_QUALITY",
        "secondary_feature": "GK_SHOT_STOPPING",
        "description": "Top distribution quality — builds from back confidently",
    },
}

POSITION_ARCHETYPES: Dict[str, Dict] = {
    "CM": CM_ARCHETYPES,
    "CB": CB_ARCHETYPES,
    "FB": FB_ARCHETYPES,
    "ST": ST_ARCHETYPES,
    "GK": GK_ARCHETYPES,
}


class PlayerArchetypeClusterer:
    """
    Clusters players within each position into football archetypes using K-Means.

    The workflow:
    1. Standardise features (preserve true variance for distance calculations)
    2. Find optimal cluster count (2-8) via elbow + silhouette
    3. Fit K-Means with n_init=20 for stability
    4. Assign archetype labels by inspecting centroid feature rankings
    5. Find similar players via Euclidean distance in feature space

    Stores fitted scalers and models per position for reuse.
    """

    def __init__(self) -> None:
        self.scalers: Dict[str, StandardScaler] = {}
        self.models: Dict[str, KMeans] = {}
        self.feature_arrays: Dict[str, np.ndarray] = {}

    def prepare_features(
        self, df: pd.DataFrame, position: str
    ) -> np.ndarray:
        """
        Select and standardise composite features for clustering.

        StandardScaler is used here (not MinMaxScaler) because:
        - Clustering operates on Euclidean distance
        - We need true relative variance preserved so rare extreme values
          (a CB with exceptional BRS) correctly influence cluster geometry
        - MinMax would artificially equate a feature spanning 0.1-50 with
          one spanning 0.1-0.8 by compressing both to [0,1]

        Parameters
        ----------
        df : pd.DataFrame
            Position-specific DataFrame with composite features.
        position : str
            Position label for feature selection.

        Returns
        -------
        np.ndarray
            Standardised feature matrix, shape (n_players, n_features).
        """
        features = CLUSTER_FEATURES.get(position, [])
        available = [f for f in features if f in df.columns]
        if not available:
            logger.error("No cluster features found for position %s", position)
            return np.array([])

        X = df[available].fillna(df[available].median()).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[position] = scaler
        self.feature_arrays[position] = X_scaled
        logger.info("Prepared %d features for %s clustering", len(available), position)
        return X_scaled

    def find_optimal_k(
        self,
        X: np.ndarray,
        k_range: Tuple[int, int] = (2, 8),
        position: str = "unknown",
    ) -> int:
        """
        Find optimal number of clusters using both elbow method and silhouette score.

        Decision rule:
        - Compute silhouette score for each k in range
        - Compute inertia (within-cluster sum of squares) for elbow
        - Return k with highest silhouette score, capped at 6
          (more than 6 archetypes per position is not useful for scouting
          communication — it fragments the taxonomy beyond practical use)

        Parameters
        ----------
        X : np.ndarray
            Standardised feature matrix.
        k_range : tuple
            (min_k, max_k) inclusive range to test.
        position : str
            Position label (for logging only).

        Returns
        -------
        int
            Optimal k.
        """
        min_k, max_k = k_range
        # Need enough players for meaningful clusters
        max_k = min(max_k, len(X) // 5, 8)
        min_k = max(2, min_k)

        if max_k < min_k:
            logger.warning("Not enough players for clustering %s, using k=2", position)
            return 2

        inertias = []
        silhouettes = []
        ks = list(range(min_k, max_k + 1))

        for k in ks:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(X)
            inertias.append(km.inertia_)
            if k > 1 and len(set(labels)) > 1:
                sil = silhouette_score(X, labels, sample_size=min(500, len(X)))
                silhouettes.append(sil)
            else:
                silhouettes.append(-1.0)

        best_k_idx = int(np.argmax(silhouettes))
        best_k = ks[best_k_idx]
        best_k = min(best_k, 6)  # cap at 6: more than 6 archetypes loses scouting utility

        logger.info(
            "Optimal k for %s: %d (silhouette=%.3f)",
            position, best_k, silhouettes[best_k_idx]
        )
        return best_k

    def fit_kmeans(
        self, X: np.ndarray, k: int, seed: int = 42
    ) -> KMeans:
        """
        Fit K-Means with n_init=20 for stability.

        n_init=20 means K-Means is run 20 times with different centroid
        initialisations. The solution with the lowest inertia is kept.
        This is essential for stability — with fewer runs, results can
        differ between executions as K-Means can converge to local minima.
        n_init=20 is the recommended value in the sklearn documentation
        for production use.

        Parameters
        ----------
        X : np.ndarray
            Standardised feature matrix.
        k : int
            Number of clusters.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        KMeans
            Fitted sklearn KMeans model.
        """
        km = KMeans(n_clusters=k, n_init=20, random_state=seed)
        km.fit(X)
        logger.info("K-Means fitted: k=%d, inertia=%.2f", k, km.inertia_)
        return km

    def label_archetypes(
        self, kmeans: KMeans, position: str, feature_names: Optional[List[str]] = None
    ) -> Dict[int, str]:
        """
        Assign football archetype labels to cluster IDs by inspecting centroids.

        The assignment logic is general: for each archetype definition, identify
        the cluster whose centroid ranks HIGHEST on the primary feature. This
        avoids hardcoding cluster numbers (which change across runs) and instead
        uses the football semantics of each cluster's dominant characteristic.

        For positions with more clusters than archetypes, remaining clusters
        are labelled as "{Position} Type {n}".

        Parameters
        ----------
        kmeans : KMeans
            Fitted K-Means model.
        position : str
            Position label.
        feature_names : list of str, optional
            Feature names corresponding to centroid columns. If None, uses
            CLUSTER_FEATURES[position].

        Returns
        -------
        dict
            Cluster ID → archetype label string.
        """
        if feature_names is None:
            feature_names = CLUSTER_FEATURES.get(position, [])

        centroids = kmeans.cluster_centers_
        n_clusters = len(centroids)
        archetypes_def = POSITION_ARCHETYPES.get(position, {})

        if not archetypes_def:
            return {i: f"{position} Type {i+1}" for i in range(n_clusters)}

        # Build feature index map
        feat_idx = {feat: i for i, feat in enumerate(feature_names)}
        assigned_clusters = {}
        used_cluster_ids = set()
        used_archetype_names = set()

        # Assign archetypes greedily by primary feature ranking
        for archetype_name, definition in archetypes_def.items():
            primary_feat = definition["primary_feature"]
            if primary_feat not in feat_idx:
                continue
            p_idx = feat_idx[primary_feat]

            # Rank clusters by primary feature value, skip already assigned
            ranking = sorted(
                [i for i in range(n_clusters) if i not in used_cluster_ids],
                key=lambda i: centroids[i][p_idx],
                reverse=True,
            )
            if not ranking:
                break

            best_cluster = ranking[0]
            assigned_clusters[best_cluster] = archetype_name
            used_cluster_ids.add(best_cluster)
            used_archetype_names.add(archetype_name)

            if len(used_cluster_ids) >= n_clusters:
                break

        # Fill remaining clusters
        label_map = {}
        fallback_n = 1
        for i in range(n_clusters):
            if i in assigned_clusters:
                label_map[i] = assigned_clusters[i]
            else:
                label_map[i] = f"{position} Profile {fallback_n}"
                fallback_n += 1

        logger.info("Archetype labels for %s: %s", position, label_map)
        return label_map

    def add_archetype_column(
        self, df: pd.DataFrame, position: str
    ) -> pd.DataFrame:
        """
        Fit clusters and add 'cluster_id' and 'archetype_label' columns to df.

        Parameters
        ----------
        df : pd.DataFrame
            Position-specific DataFrame with composite features.
        position : str
            Position label.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with 'cluster_id' and 'archetype_label' added.
        """
        df = df.copy()
        features = CLUSTER_FEATURES.get(position, [])
        available = [f for f in features if f in df.columns]

        if len(df) < 10:
            logger.warning(
                "Too few players (%d) for meaningful clustering in %s", len(df), position
            )
            df["cluster_id"] = 0
            df["archetype_label"] = f"{position} Player"
            return df

        X = self.prepare_features(df, position)
        optimal_k = self.find_optimal_k(X, position=position)
        km = self.fit_kmeans(X, optimal_k)
        self.models[position] = km

        label_map = self.label_archetypes(km, position, feature_names=available)
        df["cluster_id"] = km.labels_
        df["archetype_label"] = df["cluster_id"].map(label_map)

        logger.info(
            "Archetypes for %s: %s",
            position,
            df["archetype_label"].value_counts().to_dict()
        )
        return df

    def get_similar_players(
        self, player_name: str, df: pd.DataFrame, n: int = 5
    ) -> List[str]:
        """
        Return the n most similar players by Euclidean distance in feature space.

        Algorithm:
        1. Prioritise players in the same cluster (same archetype)
        2. If fewer than n same-cluster players exist, expand to all position players
        3. Distance is computed in standardised feature space (same as clustering)

        The "similar players" concept mirrors what scouts call "players in the same
        mould". It is not opinion — it is the mathematical nearest neighbour in
        the composite-feature space that defines playing style.

        Parameters
        ----------
        player_name : str
            Player to find similars for.
        df : pd.DataFrame
            Position-specific scored DataFrame with composite features.
        n : int
            Number of similar players to return.

        Returns
        -------
        list of str
            Player names sorted by similarity (most similar first).
        """
        position = df["position"].iloc[0] if len(df) > 0 else None
        if position is None or position not in self.scalers:
            logger.warning("No scaler found for position %s", position)
            return []

        features = CLUSTER_FEATURES.get(position, [])
        available = [f for f in features if f in df.columns]
        df_filled = df[available].fillna(df[available].median())
        X = self.scalers[position].transform(df_filled.values)

        mask = df["player_name"].str.lower().str.contains(player_name.lower(), na=False)
        if not mask.any():
            logger.warning("Player '%s' not found", player_name)
            return []

        target_idx = df[mask].index[0]
        target_pos_idx = df.index.get_loc(target_idx)
        target_vec = X[target_pos_idx]

        # Compute distances to all other players
        diffs = X - target_vec
        distances = np.linalg.norm(diffs, axis=1)
        distances[target_pos_idx] = np.inf  # exclude self

        # Prioritise same-cluster players
        if "cluster_id" in df.columns:
            target_cluster = df.iloc[target_pos_idx].get("cluster_id", -1)
            same_cluster_mask = df["cluster_id"].values == target_cluster
            same_cluster_mask[target_pos_idx] = False

            same_cluster_dists = np.where(same_cluster_mask, distances, np.inf)
            same_cluster_order = np.argsort(same_cluster_dists)
            same_cluster_names = [
                df.iloc[i]["player_name"]
                for i in same_cluster_order
                if same_cluster_mask[i]
            ][:n]

            if len(same_cluster_names) >= n:
                return same_cluster_names[:n]

            # Fill remainder from all players
            remaining_needed = n - len(same_cluster_names)
            all_order = np.argsort(distances)
            all_names = [
                df.iloc[i]["player_name"]
                for i in all_order
                if df.iloc[i]["player_name"] not in same_cluster_names
                and i != target_pos_idx
            ][:remaining_needed]

            return same_cluster_names + all_names

        all_order = np.argsort(distances)
        return [df.iloc[i]["player_name"] for i in all_order[:n]]

    def run(
        self, position_dfs: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Apply archetype clustering to all position DataFrames.

        Parameters
        ----------
        position_dfs : dict
            Scored position DataFrames from ImpactScorer.

        Returns
        -------
        dict
            Same structure with 'cluster_id' and 'archetype_label' added.
        """
        result = {}
        for pos, df in position_dfs.items():
            logger.info("Clustering %s players", pos)
            result[pos] = self.add_archetype_column(df, pos)
        logger.info("Clustering complete for all positions")
        return result
