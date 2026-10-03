"""
⚽ Premier League Player Scouting Dashboard

Streamlit app for exploring player clusters, finding similar players,
and comparing player radar charts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from model import find_similar_players, get_player_percentiles

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title=config.STREAMLIT_PAGE_TITLE,
    page_icon=config.STREAMLIT_PAGE_ICON,
    layout=config.STREAMLIT_LAYOUT,
)

# ============================================================
# Stat Glossary Definitions
# ============================================================
STAT_GLOSSARY = {
    "goals_per90": "Goals scored per 90 minutes played.",
    "assists_per90": "Assists provided per 90 minutes played.",
    "xg_per90": "Expected Goals (xG) per 90 mins — measures quality of scoring opportunities generated.",
    "xa_per90": "Expected Assists (xA) per 90 mins — measures quality of passes leading to shots.",
    "shots_per90": "Total shot attempts per 90 minutes.",
    "shots_on_target_per90": "Shot attempts on target per 90 minutes.",
    "passes_completed_per90": "Completed passes per 90 minutes.",
    "pass_completion_pct": "Percentage of total passes successfully completed.",
    "key_passes_per90": "Key Passes (KP) per 90 mins — passes leading directly to a shot attempt.",
    "progressive_passes_per90": "Progressive Passes per 90 mins — passes advancing ball >= 10 yards towards goal.",
    "progressive_carries_per90": "Progressive Carries per 90 mins — carries advancing ball >= 10 yards towards goal.",
    "successful_dribbles_per90": "Successful dribbles past opponents per 90 minutes.",
    "tackles_per90": "Successful tackles per 90 minutes.",
    "interceptions_per90": "Passes intercepted per 90 minutes.",
    "blocks_per90": "Shots and passes blocked per 90 minutes.",
    "aerials_won_per90": "Aerial duels won per 90 minutes.",
    "sca_per90": "Shot-Creating Actions (SCA) per 90 mins — offensive actions leading to a shot.",
    "gca_per90": "Goal-Creating Actions (GCA) per 90 mins — offensive actions leading to a goal.",
    "touches_per90": "Total ball touches per 90 minutes.",
    "carries_per90": "Total ball carries per 90 minutes.",
    "saves_per90": "Goalkeeper saves per 90 minutes.",
    "save_pct": "Percentage of shots on target saved by goalkeeper.",
    "clean_sheet_pct": "Percentage of matches played with a clean sheet (0 goals conceded).",
    "goals_against_per90": "Goals conceded per 90 minutes.",
    "psxg_minus_ga": "Post-Shot Expected Goals minus Goals Allowed — measures shot-stopping quality.",
    "passes_completed_pct_gk": "Percentage of launched goalkeeper passes completed.",
    "passes_launched_pct": "Percentage of keeper passes that were launched long.",
    "crosses_stopped_pct": "Percentage of opponent crosses intercepted or stopped by goalkeeper.",
    "avg_distance_defensive_actions": "Average distance (in yards) from goal line of goalkeeper's defensive actions."
}


# ============================================================
# Data Loading (cached)
# ============================================================
@st.cache_data
def load_data():
    """Load all processed data."""
    outfield = pd.read_csv(os.path.join(config.PROCESSED_DIR, "outfield_final.csv"))
    outfield_scaled = pd.read_csv(os.path.join(config.PROCESSED_DIR, "outfield_scaled_final.csv"))
    gk = pd.read_csv(os.path.join(config.PROCESSED_DIR, "gk_final.csv"))
    gk_scaled = pd.read_csv(os.path.join(config.PROCESSED_DIR, "gk_scaled_final.csv"))
    return outfield, outfield_scaled, gk, gk_scaled


# ============================================================
# Plotly Chart Functions
# ============================================================
def create_multi_player_radar(df: pd.DataFrame, selected_players: list[str], feature_cols: list[str]) -> go.Figure:
    """Create a compact, multi-player overlapping radar chart using Plotly."""
    fig = go.Figure()
    colors = ["#FF4B4B", "#00D4FF", "#FFC72C", "#00E676", "#AB47BC", "#FFA726"]

    for i, player_name in enumerate(selected_players):
        matching = df[df["Player"].str.contains(player_name, case=False, na=False)]
        if matching.empty:
            continue
        full_name = matching.iloc[0]["Player"]
        percentiles = get_player_percentiles(full_name, df, feature_cols)
        if percentiles is None:
            continue

        labels = [
            name.replace("_per90", "").replace("_pct", " %").replace("_", " ").title()
            for name in percentiles.index
        ]
        vals = percentiles.values.tolist()

        # Close the loop
        labels_loop = labels + [labels[0]]
        vals_loop = vals + [vals[0]]

        fig.add_trace(go.Scatterpolar(
            r=vals_loop,
            theta=labels_loop,
            fill="toself",
            name=full_name,
            line=dict(color=colors[i % len(colors)], width=2.5),
            opacity=0.35,
            hovertemplate="%{theta}: <b>%{r:.1f}th percentile</b><extra>" + full_name + "</extra>"
        ))

    fig.update_layout(
        polar=dict(
            bgcolor="#141b2d",
            radialaxis=dict(visible=True, range=[0, 100], color="#888", gridcolor="#2a364f"),
            angularaxis=dict(color="#ffffff", gridcolor="#2a364f", tickfont=dict(size=10, color="white"))
        ),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        margin=dict(l=30, r=30, t=30, b=30),
        height=420,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(color="white", size=11),
            bgcolor="rgba(14,17,23,0.8)"
        )
    )
    return fig


def create_interactive_cluster_map(df: pd.DataFrame, highlight_players: list[str] = None) -> go.Figure:
    """Create an interactive 2D PCA cluster map with region color splitting & hover tooltips."""
    fig = go.Figure()

    x = df["pca_x"].values
    y = df["pca_y"].values
    cluster_ids = df["cluster_id"].values

    # Grid for background region color splitting
    grid_x = np.linspace(x.min() - 0.6, x.max() + 0.6, 100)
    grid_y = np.linspace(y.min() - 0.6, y.max() + 0.6, 100)
    xx, yy = np.meshgrid(grid_x, grid_y)

    unique_c = np.unique(cluster_ids)
    centers = np.array([[x[cluster_ids == c].mean(), y[cluster_ids == c].mean()] for c in unique_c])

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    dists = np.linalg.norm(grid_points[:, np.newaxis] - centers, axis=2)
    zz = unique_c[np.argmin(dists, axis=1)].reshape(xx.shape)

    # Background Voronoi region shading
    fig.add_trace(go.Contour(
        x=grid_x, y=grid_y, z=zz,
        showscale=False, opacity=0.18,
        contours=dict(coloring="heatmap", showlines=False),
        colorscale="Viridis",
        hoverinfo="skip"
    ))

    colors = ["#FF4B4B", "#00D4FF", "#FFC72C", "#00E676", "#AB47BC", "#FFA726", "#26C6DA"]

    # Scatter points per cluster
    for i, cluster_name in enumerate(sorted(df["cluster_name"].unique())):
        c_mask = df["cluster_name"] == cluster_name
        c_df = df[c_mask]

        hover_texts = []
        for _, row in c_df.iterrows():
            txt = (
                f"<b>{row['Player']}</b><br>"
                f"Team: {row['Squad']} | Pos: {row['Pos']}<br>"
                f"Playstyle: {row['cluster_name']}<br>"
                f"Minutes: {int(row['minutes'])}<br>"
            )
            if "goals_per90" in row:
                txt += f"Goals/90: {row['goals_per90']:.2f} | Assists/90: {row['assists_per90']:.2f}<br>"
                txt += f"xG/90: {row['xg_per90']:.2f} | Tackles/90: {row['tackles_per90']:.2f}"
            else:
                txt += f"Save %: {row['save_pct']:.1f}% | CS %: {row['clean_sheet_pct']:.1f}%"
            hover_texts.append(txt)

        fig.add_trace(go.Scatter(
            x=c_df["pca_x"],
            y=c_df["pca_y"],
            mode="markers",
            name=cluster_name,
            text=hover_texts,
            hoverinfo="text",
            marker=dict(
                size=9,
                color=colors[i % len(colors)],
                line=dict(width=0.8, color="white"),
                opacity=0.85
            )
        ))

    # Highlight selected player(s)
    if highlight_players:
        for hp in highlight_players:
            hp_mask = df["Player"].str.contains(hp, case=False, na=False)
            if hp_mask.any():
                hp_data = df[hp_mask].iloc[0]
                fig.add_trace(go.Scatter(
                    x=[hp_data["pca_x"]],
                    y=[hp_data["pca_y"]],
                    mode="markers+text",
                    name=f"⭐ {hp_data['Player']}",
                    text=[f"  <b>{hp_data['Player']}</b>"],
                    textposition="top right",
                    textfont=dict(color="white", size=12),
                    hoverinfo="skip",
                    marker=dict(
                        size=18,
                        symbol="star",
                        color="#FF0055",
                        line=dict(width=2, color="white")
                    )
                ))

    fig.update_layout(
        title=dict(text="<b>Player Playstyle Map (PCA Projection & Regions)</b>", font=dict(size=15, color="white")),
        xaxis=dict(title="PCA Axis 1", color="white", gridcolor="#2a364f", showgrid=True),
        yaxis=dict(title="PCA Axis 2", color="white", gridcolor="#2a364f", showgrid=True),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#141b2d",
        height=520,
        margin=dict(l=30, r=30, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(color="white", size=10),
            bgcolor="rgba(14,17,23,0.8)"
        )
    )

    return fig


def create_comparison_bar(player_name: str, similar_player_name: str,
                          df: pd.DataFrame, feature_cols: list[str]) -> go.Figure:
    """Side-by-side Plotly bar chart comparing two players."""
    p1_mask = df["Player"].str.contains(player_name, case=False, na=False)
    p2_mask = df["Player"].str.contains(similar_player_name, case=False, na=False)

    if not p1_mask.any() or not p2_mask.any():
        return go.Figure()

    p1 = df[p1_mask].iloc[0]
    p2 = df[p2_mask].iloc[0]

    available = [f for f in feature_cols if f in df.columns][:10]
    labels = [
        name.replace("_per90", "/90").replace("_pct", "%").replace("_", " ").title()
        for name in available
    ]

    p1_vals = [p1[f] for f in available]
    p2_vals = [p2[f] for f in available]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=p1_vals, name=p1["Player"], marker_color="#FF4B4B"))
    fig.add_trace(go.Bar(x=labels, y=p2_vals, name=p2["Player"], marker_color="#00D4FF"))

    fig.update_layout(
        barmode="group",
        title=dict(text=f"<b>Metric Comparison: {p1['Player']} vs {p2['Player']}</b>", font=dict(size=14, color="white")),
        xaxis=dict(color="white", tickangle=-30),
        yaxis=dict(color="white", gridcolor="#2a364f"),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#141b2d",
        height=380,
        margin=dict(l=30, r=30, t=50, b=40),
        legend=dict(font=dict(color="white"))
    )
    return fig


# ============================================================
# Main App
# ============================================================
def main():
    # --- Header ---
    st.title("⚽ Premier League Player Scout")
    st.markdown("*Find similar players, compare overlapping radars, and explore playstyle clusters — PL 2025-26*")
    st.divider()

    # --- Load Data ---
    try:
        outfield, outfield_scaled, gk, gk_scaled = load_data()
    except Exception:
        st.error("⚠️ Processed data files not found. Click below to run the pipeline automatically:")
        if st.button("🚀 Run Pipeline Now"):
            with st.spinner("Fetching Premier League 2025-26 data and building ML models..."):
                from run_pipeline import run_pipeline
                run_pipeline()
                st.cache_data.clear()
                st.rerun()
        return

    # --- Sidebar ---
    st.sidebar.header("🔍 Player Search & Comparison")

    player_type = st.sidebar.radio("Player Type", ["Outfield", "Goalkeeper"])

    if player_type == "Outfield":
        df = outfield
        df_scaled = outfield_scaled
        feature_cols = [c for c in config.OUTFIELD_FEATURES if c in df.columns]
    else:
        df = gk
        df_scaled = gk_scaled
        feature_cols = [c for c in config.GK_FEATURES if c in df.columns]

    # Full list of all players (sorted alphabetically)
    all_players = sorted(df["Player"].dropna().unique().tolist())

    # --- PRIMARY GENERAL DROPDOWN (PROMINENT AT TOP) ---
    selected_player = st.sidebar.selectbox(
        "👤 Select Player (All Players)",
        all_players,
        index=0,
        help="Search or pick any player from all 330+ Premier League players. You can type directly into this box to search!"
    )

    # Multi-player selection for radar overlay
    other_players = [p for p in all_players if p != selected_player]
    compare_players_selected = st.sidebar.multiselect(
        "⚔️ Compare Radar With (up to 3)",
        other_players,
        max_selections=3,
        help="Select additional players to overlap on the radar chart"
    )

    n_similar = st.sidebar.slider("Number of similar players to show", 3, 20, 10)

    # --- OPTIONAL FILTERS EXPANDER ---
    st.sidebar.divider()
    with st.sidebar.expander("🎛️ Optional Filters (Team, Position, Cluster, Minutes)"):
        all_squads = ["All Teams"] + sorted(df["Squad"].dropna().unique().tolist())
        selected_squad = st.selectbox("🏟️ Filter by Team", all_squads)

        if player_type == "Outfield" and "Pos" in df.columns:
            all_positions = ["All Positions"] + sorted(df["Pos"].dropna().unique().tolist())
            selected_pos = st.selectbox("📌 Filter by Position", all_positions)
        else:
            selected_pos = "All Positions"

        all_clusters = ["All Clusters"] + sorted(df["cluster_name"].dropna().unique().tolist())
        selected_cluster = st.selectbox("🏷️ Filter by Cluster", all_clusters)

        min_mins_val = int(df["minutes"].min()) if "minutes" in df.columns else 0
        max_mins_val = int(df["minutes"].max()) if "minutes" in df.columns else 3000
        min_minutes_filter = st.slider(
            "⏱️ Min Minutes Played",
            min_value=min_mins_val,
            max_value=max_mins_val,
            value=min_mins_val,
            step=45
        )

    # Sidebar Quick Stat Guide Expander
    with st.sidebar.expander("📖 Quick Stat Guide"):
        st.caption("Key Metrics Defined:")
        st.markdown("**xG / 90**: Expected Goals per 90 mins")
        st.markdown("**xA / 90**: Expected Assists per 90 mins")
        st.markdown("**SCA / 90**: Shot-Creating Actions per 90 mins")
        st.markdown("**GCA / 90**: Goal-Creating Actions per 90 mins")
        st.markdown("**KP / 90**: Key Passes leading directly to shots")
        st.markdown("**PrgP / 90**: Progressive Passes >= 10 yards forward")
        st.markdown("**PrgC / 90**: Progressive Carries >= 10 yards forward")
        st.markdown("**PSxG +/-**: Post-shot xG minus goals allowed")

    st.sidebar.divider()
    if st.sidebar.button("🔄 Update Data & Re-run Pipeline"):
        with st.spinner("Updating Premier League stats & retraining models..."):
            from run_pipeline import run_pipeline
            run_pipeline()
            st.cache_data.clear()
            st.rerun()

    # --- Main Content ---
    player_mask = df["Player"] == selected_player
    if not player_mask.any():
        st.warning("Player not found.")
        return

    player_data = df[player_mask].iloc[0]

    # ---- Player Summary Card ----
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.subheader(f"🧑 {player_data['Player']}")
        st.markdown(f"**Team:** {player_data.get('Squad', 'N/A')}")
        st.markdown(f"**Position:** {player_data.get('Pos', 'N/A')}")
        st.markdown(f"**Minutes:** {player_data.get('minutes', 0):,.0f}")
    with col2:
        st.subheader(f"🏷️ {player_data.get('cluster_name', 'N/A')}")
        st.markdown(f"**90s Played:** {player_data.get('90s_played', 0):.1f}")
        if player_type == "Outfield":
            st.markdown(f"**Goals/90:** {player_data.get('goals_per90', 0):.2f}")
            st.markdown(f"**Assists/90:** {player_data.get('assists_per90', 0):.2f}")
    with col3:
        if player_type == "Outfield":
            st.metric("xG/90", f"{player_data.get('xg_per90', 0):.2f}", help=STAT_GLOSSARY.get("xg_per90"))
            st.metric("SCA/90", f"{player_data.get('sca_per90', 0):.2f}", help=STAT_GLOSSARY.get("sca_per90"))
            st.metric("Tackles/90", f"{player_data.get('tackles_per90', 0):.2f}", help=STAT_GLOSSARY.get("tackles_per90"))
        else:
            st.metric("Save %", f"{player_data.get('save_pct', 0):.1f}%", help=STAT_GLOSSARY.get("save_pct"))
            st.metric("CS %", f"{player_data.get('clean_sheet_pct', 0):.1f}%", help=STAT_GLOSSARY.get("clean_sheet_pct"))

    st.divider()

    # ---- Tabs ----
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Radar Comparison", "👥 Similar Players", "🗺️ Cluster Map", "📖 Full Stats & Guide"
    ])

    # Combine primary player + compare players for radar
    radar_players = [selected_player] + compare_players_selected

    # ---- Tab 1: Overlapping Radar Chart ----
    with tab1:
        st.subheader("📊 Percentile Radar Comparison")
        st.caption("Overlapping radar charts of selected players across all per-90 percentile metrics.")

        col_radar, col_legend = st.columns([3, 1])
        with col_radar:
            fig_radar = create_multi_player_radar(df, radar_players, feature_cols)
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_legend:
            st.markdown("### 🧑 Selected Players")
            for idx, p_item in enumerate(radar_players):
                colors = ["🔴 Primary", "🔵 Compare 1", "🟡 Compare 2", "🟢 Compare 3"]
                st.markdown(f"**{colors[idx % 4]}**: {p_item}")

            st.divider()
            st.caption("💡 *Percentiles range from 0 to 100 relative to all players in the league.*")

    # ---- Tab 2: Similar Players ----
    with tab2:
        st.subheader(f"Top {n_similar} Players Similar to {selected_player}")

        similar = find_similar_players(
            selected_player, df, df_scaled, feature_cols, n=n_similar
        )

        if similar is not None and len(similar) > 0:
            if selected_squad != "All Teams":
                similar = similar[similar["Squad"] == selected_squad]
            if selected_pos != "All Positions" and "Pos" in similar.columns:
                similar = similar[similar["Pos"] == selected_pos]
            if selected_cluster != "All Clusters":
                similar = similar[similar["cluster_name"] == selected_cluster]
            if "minutes" in similar.columns:
                similar = similar[similar["minutes"] >= min_minutes_filter]

            display_df = similar[["Player", "Squad", "Pos", "cluster_name", "similarity_pct"]].copy()
            display_df.columns = ["Player", "Team", "Position", "Playstyle", "Match %"]
            display_df.index = range(1, len(display_df) + 1)
            display_df.index.name = "Rank"

            st.dataframe(
                display_df.style.format({"Match %": "{:.1f}%"}),
                use_container_width=True
            )

            st.divider()
            compare_player = st.selectbox(
                "Bar comparison with top match:", similar["Player"].tolist()
            )
            if compare_player:
                fig_bar = create_comparison_bar(selected_player, compare_player, df, feature_cols)
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No similar players found.")

    # ---- Tab 3: Interactive Cluster Map ----
    with tab3:
        st.subheader("🗺️ Interactive Player Cluster Map")
        st.caption("Hover over dots to see player details. Background colors show playstyle regional boundaries.")

        fig_cluster = create_interactive_cluster_map(df, highlight_players=radar_players)
        st.plotly_chart(fig_cluster, use_container_width=True)

        st.divider()
        st.subheader("Cluster Summary Breakdown")
        cluster_summary = (
            df.groupby("cluster_name")
            .agg(
                Players=("Player", "count"),
                Avg_Minutes=("minutes", "mean"),
            )
            .round(0)
            .sort_values("Players", ascending=False)
        )
        st.dataframe(cluster_summary, use_container_width=True)

    # ---- Tab 4: Full Stats & Complete Stat Guide ----
    with tab4:
        st.subheader(f"📋 Full Stats — {selected_player}")

        player_stats = df[player_mask][feature_cols].T
        player_stats.columns = [selected_player]

        # Build combined table with values + descriptions
        stat_rows = []
        for feat in feature_cols:
            val = player_stats.loc[feat, selected_player] if feat in player_stats.index else 0.0
            label = feat.replace("_per90", "/90").replace("_pct", " %").replace("_", " ").title()
            desc = STAT_GLOSSARY.get(feat, "Standard player performance metric.")
            stat_rows.append({"Metric": label, "Value": f"{val:.2f}", "Definition & Guide": desc})

        guide_df = pd.DataFrame(stat_rows)
        st.dataframe(guide_df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("📖 Complete Premier League Stat Glossary")
        for key, desc in STAT_GLOSSARY.items():
            l_name = key.replace("_per90", "/90").replace("_pct", " %").replace("_", " ").title()
            st.markdown(f"**{l_name}**: {desc}")

    # ---- Footer ----
    st.divider()
    st.caption("Data: Premier League 2025-26 | Built with Streamlit, Plotly & scikit-learn")


if __name__ == "__main__":
    main()
