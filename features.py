"""
Feature Engineering Pipeline.

Takes raw scraped FBref data and produces clean, normalized per-90 features
ready for clustering.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import config


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the raw scraped CSVs."""
    outfield = pd.read_csv(os.path.join(config.RAW_DIR, "outfield_raw.csv"))
    gk = pd.read_csv(os.path.join(config.RAW_DIR, "gk_raw.csv"))
    print(f"Loaded outfield: {outfield.shape}, GK: {gk.shape}")
    return outfield, gk


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find the first matching column name from a list of candidates."""
    for c in candidates:
        # Exact match
        if c in df.columns:
            return c
        # Case-insensitive match
        for col in df.columns:
            if col.lower().strip() == c.lower().strip():
                return col
    return None


def _safe_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    """Convert a column to numeric, coercing errors to NaN."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def process_outfield(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process outfield player data:
    1. Filter by minimum minutes
    2. Compute per-90 features
    3. Handle missing values
    """
    print("\n--- Processing Outfield Players ---")

    # Find the minutes column (FBref uses various names)
    min_col = _find_column(df, ["Min", "Playing Time_Min", "90s", "Minutes"])
    if min_col is None:
        # Try to find any column containing 'Min'
        min_candidates = [c for c in df.columns if "min" in c.lower()]
        if min_candidates:
            min_col = min_candidates[0]
        else:
            raise ValueError(f"Cannot find minutes column! Available: {df.columns.tolist()}")

    print(f"  Using minutes column: '{min_col}'")

    # Convert minutes to numeric (handle commas like "1,234")
    df["minutes"] = pd.to_numeric(df[min_col].astype(str).str.replace(",", ""), errors="coerce")

    # Filter by minimum minutes
    df = df[df["minutes"] >= config.MIN_MINUTES_PLAYED].copy()
    print(f"  After filtering ({config.MIN_MINUTES_PLAYED}+ min): {len(df)} players")

    # Calculate 90s played
    df["90s_played"] = df["minutes"] / 90.0

    # --- Build per-90 features ---
    # We'll try multiple possible column names for each stat since FBref
    # column names can vary depending on the season/page structure

    feature_mapping = {
        # Attacking
        "goals_per90": (["Performance_Gls", "Gls", "Standard_Gls"], "per90"),
        "assists_per90": (["Performance_Ast", "Ast", "Standard_Ast"], "per90"),
        "xg_per90": (["Expected_xG", "xG"], "per90"),
        "xa_per90": (["Expected_xAG", "xAG", "xA"], "per90"),
        # Shooting
        "shots_per90": (["Standard_Sh", "Sh"], "per90"),
        "shots_on_target_per90": (["Standard_SoT", "SoT"], "per90"),
        # Passing
        "passes_completed_per90": (["Total_Cmp", "Cmp"], "per90"),
        "pass_completion_pct": (["Total_Cmp%", "Cmp%"], "direct"),
        "key_passes_per90": (["KP"], "per90"),
        "progressive_passes_per90": (["PrgP", "Progression_PrgP"], "per90"),
        # Carries / Dribbles
        "progressive_carries_per90": (["PrgC", "Carries_PrgC", "Progression_PrgC"], "per90"),
        "successful_dribbles_per90": (["Take-Ons_Succ", "Dribbles_Succ", "Succ"], "per90"),
        # Defense
        "tackles_per90": (["Tackles_Tkl", "Tkl"], "per90"),
        "interceptions_per90": (["Int"], "per90"),
        "blocks_per90": (["Blocks_Blocks", "Blocks"], "per90"),
        "aerials_won_per90": (["Aerial Duels_Won", "Won"], "per90"),
        # Creation
        "sca_per90": (["SCA_SCA", "SCA"], "per90"),
        "gca_per90": (["GCA_GCA", "GCA"], "per90"),
        # Possession
        "touches_per90": (["Touches_Touches", "Touches"], "per90"),
        "carries_per90": (["Carries_Carries", "Carries"], "per90"),
    }

    for feat_name, (candidates, method) in feature_mapping.items():
        col = _find_column(df, candidates)
        if col is not None:
            values = _safe_numeric(df, col)
            if method == "per90":
                df[feat_name] = values / df["90s_played"]
            else:
                df[feat_name] = values
        else:
            print(f"  WARNING: Could not find column for '{feat_name}' (tried: {candidates})")
            df[feat_name] = np.nan

    # --- Build output DataFrame ---
    id_cols = [c for c in ["Player", "Squad", "Pos", "Age", "Nation", "minutes", "90s_played"] if c in df.columns]
    feature_cols = list(feature_mapping.keys())

    # Keep only columns that exist
    available_features = [f for f in feature_cols if f in df.columns and df[f].notna().any()]
    print(f"  Available features: {len(available_features)}/{len(feature_cols)}")

    result = df[id_cols + available_features].copy()

    # Fill remaining NaN with 0 (if a player has no shots, that's 0, not missing)
    result[available_features] = result[available_features].fillna(0)

    # Replace infinities with 0
    result[available_features] = result[available_features].replace([np.inf, -np.inf], 0)

    print(f"  Final outfield shape: {result.shape}")
    return result


def process_gk(df: pd.DataFrame) -> pd.DataFrame:
    """Process goalkeeper data into clean features."""
    print("\n--- Processing Goalkeepers ---")

    # Find minutes column
    min_col = _find_column(df, ["Min", "Playing Time_Min", "90s", "Minutes"])
    if min_col is None:
        min_candidates = [c for c in df.columns if "min" in c.lower()]
        min_col = min_candidates[0] if min_candidates else None

    if min_col:
        df["minutes"] = pd.to_numeric(df[min_col].astype(str).str.replace(",", ""), errors="coerce")
        df = df[df["minutes"] >= config.MIN_MINUTES_PLAYED].copy()
    else:
        df["minutes"] = np.nan

    df["90s_played"] = df["minutes"] / 90.0

    # GK feature mapping — exact FBref column names (multi-header flattened)
    gk_feature_mapping = {
        "saves_per90": (["Performance_Saves", "Saves"], "per90"),
        "save_pct": (["Performance_Save%", "Save%"], "direct"),
        "clean_sheet_pct": (["Performance_CS%", "CS%"], "direct"),
        "goals_against_per90": (["Performance_GA", "GA"], "per90"),
        "psxg_minus_ga": (["Shot Stopping_PSxG+/-", "PSxG+/-", "Expected_PSxG+/-"], "direct"),
        "passes_completed_pct_gk": (["Launched_Cmp%", "Cmp%"], "direct"),
        "passes_launched_pct": (["Launch%", "Launched_Launch%"], "direct"),
        "crosses_stopped_pct": (["Crosses_Stp%", "Stp%"], "direct"),
        "avg_distance_defensive_actions": (["Sweeper_AvgDist", "#OPA_AvgDist", "AvgDist"], "direct"),
    }

    for feat_name, (candidates, method) in gk_feature_mapping.items():
        col = _find_column(df, candidates)
        if col is not None:
            values = _safe_numeric(df, col)
            if method == "per90":
                df[feat_name] = values / df["90s_played"]
            else:
                df[feat_name] = values
        else:
            print(f"  WARNING: Could not find column for '{feat_name}'")
            df[feat_name] = np.nan

    id_cols = [c for c in ["Player", "Squad", "Pos", "Age", "Nation", "minutes", "90s_played"] if c in df.columns]
    feature_cols = list(gk_feature_mapping.keys())
    available_features = [f for f in feature_cols if f in df.columns and df[f].notna().any()]

    result = df[[c for c in id_cols if c in df.columns] + available_features].copy()
    result[available_features] = result[available_features].fillna(0)
    result[available_features] = result[available_features].replace([np.inf, -np.inf], 0)

    print(f"  Final GK shape: {result.shape}")
    return result


def scale_features(df: pd.DataFrame, feature_cols: list[str], label: str) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Standardize features (zero mean, unit variance).
    Returns the scaled DataFrame and the fitted scaler.
    """
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[feature_cols])
    scaled_df = df.copy()
    scaled_df[feature_cols] = scaled_values

    # Save scaler
    scaler_path = os.path.join(config.MODELS_DIR, f"scaler_{label}.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"  Saved scaler: {scaler_path}")

    return scaled_df, scaler


def main():
    """Run the full feature engineering pipeline."""
    print("🔄 Starting feature engineering pipeline...\n")

    outfield_raw, gk_raw = load_raw_data()

    # Process outfield players
    outfield = process_outfield(outfield_raw)
    outfield_features = [c for c in config.OUTFIELD_FEATURES if c in outfield.columns]
    outfield_scaled, _ = scale_features(outfield, outfield_features, "outfield")

    # Save processed data
    outfield.to_csv(os.path.join(config.PROCESSED_DIR, "outfield_processed.csv"), index=False)
    outfield_scaled.to_csv(os.path.join(config.PROCESSED_DIR, "outfield_scaled.csv"), index=False)

    # Process goalkeepers
    gk = process_gk(gk_raw)
    gk_features = [c for c in config.GK_FEATURES if c in gk.columns]
    gk_scaled, _ = scale_features(gk, gk_features, "gk")

    gk.to_csv(os.path.join(config.PROCESSED_DIR, "gk_processed.csv"), index=False)
    gk_scaled.to_csv(os.path.join(config.PROCESSED_DIR, "gk_scaled.csv"), index=False)

    print("\n🎉 Feature engineering complete!")
    print(f"  Outfield: {len(outfield)} players, {len(outfield_features)} features")
    print(f"  GK: {len(gk)} players, {len(gk_features)} features")


if __name__ == "__main__":
    main()
