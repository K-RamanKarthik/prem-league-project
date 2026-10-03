"""
ML Pipeline: K-Means Clustering + Nearest Neighbor Similarity Search.

Clusters players into playstyle groups and provides functionality
to find the most similar players to any given player.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import joblib

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import config


def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load processed and scaled data for both outfield and GK."""
    outfield = pd.read_csv(os.path.join(config.PROCESSED_DIR, "outfield_processed.csv"))
    outfield_scaled = pd.read_csv(os.path.join(config.PROCESSED_DIR, "outfield_scaled.csv"))
    gk = pd.read_csv(os.path.join(config.PROCESSED_DIR, "gk_processed.csv"))
    gk_scaled = pd.read_csv(os.path.join(config.PROCESSED_DIR, "gk_scaled.csv"))
    return outfield, outfield_scaled, gk, gk_scaled


def run_kmeans(
    df_scaled: pd.DataFrame,
    feature_cols: list[str],
    n_clusters: int,
    label: str,
) -> tuple[KMeans, np.ndarray]:
    """
    Run K-Means clustering on scaled features.
    Returns the fitted model and cluster labels.
    """
    print(f"\n  Running K-Means (k={n_clusters}) on {label}...")

    X = df_scaled[feature_cols].values
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=config.RANDOM_STATE,
        n_init=20,
        max_iter=500,
    )
    clusters = kmeans.fit_predict(X)

    # Print cluster distribution
    unique, counts = np.unique(clusters, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    Cluster {u}: {c} players")

    # Save model
    model_path = os.path.join(config.MODELS_DIR, f"kmeans_{label}.pkl")
    joblib.dump(kmeans, model_path)
    print(f"  Saved model: {model_path}")

    return kmeans, clusters


def run_pca(
    df_scaled: pd.DataFrame,
    feature_cols: list[str],
    label: str,
    n_components: int = 2,
) -> tuple[PCA, np.ndarray]:
    """
    Run PCA for 2D visualization of clusters.
    """
    X = df_scaled[feature_cols].values
    pca = PCA(n_components=n_components, random_state=config.RANDOM_STATE)
    coords = pca.fit_transform(X)

    explained = sum(pca.explained_variance_ratio_) * 100
    print(f"  PCA ({label}): {explained:.1f}% variance explained by {n_components} components")

    # Save PCA model
    pca_path = os.path.join(config.MODELS_DIR, f"pca_{label}.pkl")
    joblib.dump(pca, pca_path)

    return pca, coords


def assign_cluster_labels(
    df: pd.DataFrame,
    clusters: np.ndarray,
    df_scaled: pd.DataFrame,
    feature_cols: list[str],
    label_map: dict[int, str],
    kmeans: KMeans,
) -> pd.DataFrame:
    """
    Assign human-readable cluster labels based on centroid analysis.
    Automatically names clusters based on their dominant features.
    """
    df = df.copy()
    df["cluster_id"] = clusters

    # Analyze centroids to auto-generate labels
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=feature_cols)

    auto_labels = {}
    for i in range(len(centroids)):
        centroid = centroids.iloc[i]
        # Find the top 3 features for this cluster
        top_features = centroid.abs().nlargest(3).index.tolist()
        # Create a descriptive name based on dominant stats
        auto_labels[i] = _generate_cluster_name(top_features, centroid)

    print(f"\n  Auto-generated cluster labels ({label_map.__class__.__name__}):")
    for cid, name in auto_labels.items():
        count = (clusters == cid).sum()
        print(f"    Cluster {cid}: {name} ({count} players)")

    df["cluster_name"] = df["cluster_id"].map(auto_labels)
    return df


def _generate_cluster_name(top_features: list[str], centroid: pd.Series) -> str:
    """Generate a human-readable cluster name from dominant features."""
    # Mapping of feature patterns to descriptive names
    name_hints = {
        "goals": "Goal-Scoring",
        "xg": "Goal-Threat",
        "assists": "Creative",
        "xa": "Chance-Creating",
        "key_passes": "Playmaking",
        "progressive_passes": "Progressive Passing",
        "progressive_carries": "Ball-Carrying",
        "dribbles": "Dribbling",
        "tackles": "Tackling",
        "interceptions": "Intercepting",
        "blocks": "Defensive",
        "aerials": "Aerial",
        "sca": "Chance-Creating",
        "gca": "Goal-Creating",
        "touches": "High-Touch",
        "carries": "Ball-Progressing",
        "shots": "Shot-Volume",
        "pass_completion": "Possession-Based",
        "saves": "Shot-Stopping",
        "clean_sheet": "Clean-Sheet",
        "psxg": "Elite Shot-Stopping",
        "crosses_stopped": "Cross-Claiming",
        "launched": "Long-Distribution",
    }

    descriptors = []
    for feat in top_features[:2]:  # Use top 2 features for naming
        for pattern, name in name_hints.items():
            if pattern in feat.lower():
                if name not in descriptors:
                    descriptors.append(name)
                break

    if descriptors:
        return " + ".join(descriptors[:2])
    return f"Style {top_features[0]}"


def find_similar_players(
    player_name: str,
    df: pd.DataFrame,
    df_scaled: pd.DataFrame,
    feature_cols: list[str],
    n: int = config.N_SIMILAR_PLAYERS,
) -> pd.DataFrame | None:
    """
    Find the N most similar players to a given player using cosine similarity.
    Returns a DataFrame with similar players and their similarity scores.
    """
    # Find the player
    mask = df["Player"].str.contains(player_name, case=False, na=False)
    if mask.sum() == 0:
        return None

    # Get the player's index (first match)
    player_idx = mask.idxmax()
    player_features = df_scaled.loc[player_idx, feature_cols].values.reshape(1, -1)

    # Calculate cosine similarity with all other players
    all_features = df_scaled[feature_cols].values
    similarities = cosine_similarity(player_features, all_features)[0]

    # Create results DataFrame
    results = df[["Player", "Squad", "Pos", "cluster_name"]].copy()
    results["similarity"] = similarities
    results["similarity_pct"] = (similarities * 100).round(1)

    # Exclude the player themselves and sort by similarity
    results = results[results.index != player_idx]
    results = results.sort_values("similarity", ascending=False).head(n)

    return results.reset_index(drop=True)


def get_player_percentiles(
    player_name: str,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.Series | None:
    """
    Calculate percentile ranks for a player across all features.
    Used for radar charts.
    """
    mask = df["Player"].str.contains(player_name, case=False, na=False)
    if mask.sum() == 0:
        return None

    player_idx = mask.idxmax()

    percentiles = {}
    for col in feature_cols:
        values = df[col].dropna()
        player_val = df.loc[player_idx, col]
        percentiles[col] = (values < player_val).mean() * 100

    return pd.Series(percentiles)


def main():
    """Run the full ML pipeline."""
    print("🔄 Starting ML pipeline...\n")

    outfield, outfield_scaled, gk, gk_scaled = load_processed_data()

    # --- Outfield Clustering ---
    outfield_features = [c for c in config.OUTFIELD_FEATURES if c in outfield_scaled.columns]
    print(f"Using {len(outfield_features)} outfield features")

    kmeans_outfield, clusters_outfield = run_kmeans(
        outfield_scaled, outfield_features,
        config.N_CLUSTERS_OUTFIELD, "outfield"
    )

    pca_outfield, pca_coords_outfield = run_pca(
        outfield_scaled, outfield_features, "outfield"
    )

    outfield = assign_cluster_labels(
        outfield, clusters_outfield, outfield_scaled,
        outfield_features, config.OUTFIELD_CLUSTER_LABELS, kmeans_outfield
    )
    outfield["pca_x"] = pca_coords_outfield[:, 0]
    outfield["pca_y"] = pca_coords_outfield[:, 1]

    # --- GK Clustering ---
    gk_features = [c for c in config.GK_FEATURES if c in gk_scaled.columns]
    print(f"\nUsing {len(gk_features)} GK features")

    kmeans_gk, clusters_gk = run_kmeans(
        gk_scaled, gk_features,
        config.N_CLUSTERS_GK, "gk"
    )

    pca_gk, pca_coords_gk = run_pca(
        gk_scaled, gk_features, "gk"
    )

    gk = assign_cluster_labels(
        gk, clusters_gk, gk_scaled,
        gk_features, config.GK_CLUSTER_LABELS, kmeans_gk
    )
    gk["pca_x"] = pca_coords_gk[:, 0]
    gk["pca_y"] = pca_coords_gk[:, 1]

    # --- Save Final Data ---
    outfield.to_csv(os.path.join(config.PROCESSED_DIR, "outfield_final.csv"), index=False)
    gk.to_csv(os.path.join(config.PROCESSED_DIR, "gk_final.csv"), index=False)

    # Save scaled versions with clusters too
    outfield_scaled["cluster_id"] = clusters_outfield
    gk_scaled["cluster_id"] = clusters_gk
    outfield_scaled.to_csv(os.path.join(config.PROCESSED_DIR, "outfield_scaled_final.csv"), index=False)
    gk_scaled.to_csv(os.path.join(config.PROCESSED_DIR, "gk_scaled_final.csv"), index=False)

    print("\n🎉 ML pipeline complete!")
    print(f"  Outfield: {len(outfield)} players in {config.N_CLUSTERS_OUTFIELD} clusters")
    print(f"  GK: {len(gk)} players in {config.N_CLUSTERS_GK} clusters")

    # --- Quick Demo ---
    print("\n" + "=" * 60)
    print("DEMO: Finding similar players")
    print("=" * 60)

    # Pick a well-known player for demo
    for test_player in ["Salah", "Haaland", "Saka", "Rodri"]:
        similar = find_similar_players(test_player, outfield, outfield_scaled, outfield_features, n=5)
        if similar is not None:
            print(f"\n  Top 5 similar to '{test_player}':")
            for _, row in similar.iterrows():
                print(f"    {row['Player']:30s} ({row['Squad']:20s}) — {row['similarity_pct']}% match")


if __name__ == "__main__":
    main()
