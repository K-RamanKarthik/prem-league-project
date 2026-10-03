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

try:
    from scipy.spatial import ConvexHull
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

CLUSTER_COLORS = [
    "#FF4B4B",  # Vibrant Red
    "#00D4FF",  # Bright Cyan
    "#FFC72C",  # Warm Yellow
    "#00E676",  # Neon Green
    "#AB47BC",  # Electric Purple
    "#FFA726",  # Orange
    "#26C6DA",  # Teal
    "#EC407A",  # Pink
    "#7E57C2",  # Deep Violet
    "#78909C"   # Slate Blue
]

def hex_to_rgba(hex_str: str, opacity: float = 0.15) -> str:
    """Convert hex color string to rgba string."""
    hex_str = hex_str.lstrip('#')
    r, g, b = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {opacity})"


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
def fix_mojibake(val):
    """Fix cp1252/latin1 mojibake characters to proper UTF-8 strings."""
    if not isinstance(val, str):
        return val
    if any(c in val for c in ["Ã", "Â", "Å"]):
        try:
            return val.encode("latin1").decode("utf-8")
        except Exception:
            return val
    return val

@st.cache_data
def load_data():
    """Load all processed data with UTF-8 character decoding."""
    outfield = pd.read_csv(os.path.join(config.PROCESSED_DIR, "outfield_final.csv"), encoding="utf-8")
    outfield_scaled = pd.read_csv(os.path.join(config.PROCESSED_DIR, "outfield_scaled_final.csv"), encoding="utf-8")
    gk = pd.read_csv(os.path.join(config.PROCESSED_DIR, "gk_final.csv"), encoding="utf-8")
    gk_scaled = pd.read_csv(os.path.join(config.PROCESSED_DIR, "gk_scaled_final.csv"), encoding="utf-8")

    # Clean character encodings across all loaded datasets
    for d in [outfield, outfield_scaled, gk, gk_scaled]:
        for col in d.select_dtypes(include=["object"]).columns:
            d[col] = d[col].apply(fix_mojibake)
        if "Player" in d.columns:
            d["Player"] = d["Player"].astype(str).str.strip()

    return outfield, outfield_scaled, gk, gk_scaled


def get_sorted_players(df: pd.DataFrame) -> list[str]:
    """
    Return sorted list of player names:
    1. Active players with > 0 minutes first in alphabetical order (e.g. #1 Aaron Hickey).
    2. Players with 0 minutes second.
    3. 'Aaron Anselmino' (and any Anselmo variations) pushed explicitly to the absolute bottom of the list.
    """
    players = df["Player"].dropna().unique().tolist()

    def player_sort_key(p_name: str):
        p_lower = p_name.lower()
        if "anselm" in p_lower or "anselim" in p_lower:
            return (3, p_lower)
        p_rows = df[df["Player"] == p_name]
        mins = p_rows["minutes"].iloc[0] if not p_rows.empty and "minutes" in p_rows.columns else 0
        if mins == 0:
            return (2, p_lower)
        return (0, p_lower)

    return sorted(players, key=player_sort_key)


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
        font=dict(family="Inter, Outfit, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Arial, sans-serif", color="white"),
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


def create_interactive_cluster_map(
    df: pd.DataFrame, 
    highlight_players: list[str] = None,
    zone_style: str = "Convex Hull Bubbles",
    show_centroids: bool = True
) -> go.Figure:
    """Create a remade, high-precision 2D PCA interactive cluster map from scratch."""
    fig = go.Figure()

    if df.empty or "pca_x" not in df.columns or "pca_y" not in df.columns:
        return fig

    x = df["pca_x"].values
    y = df["pca_y"].values

    sorted_clusters = sorted(df["cluster_name"].unique())
    cluster_color_map = {c_name: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, c_name in enumerate(sorted_clusters)}

    # --- 1. Background Playstyle Region Shading ---
    if zone_style == "Convex Hull Bubbles" and HAS_SCIPY:
        for c_name in sorted_clusters:
            c_df = df[df["cluster_name"] == c_name]
            pts = c_df[["pca_x", "pca_y"]].values
            c_color = cluster_color_map[c_name]

            if len(pts) >= 3:
                try:
                    hull = ConvexHull(pts)
                    hull_pts = pts[hull.vertices]
                    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()

                    # Expand hull slightly (10%) from centroid for smooth envelopment
                    exp_x = list(cx + 1.10 * (hull_pts[:, 0] - cx))
                    exp_y = list(cy + 1.10 * (hull_pts[:, 1] - cy))

                    # Close loop
                    exp_x.append(exp_x[0])
                    exp_y.append(exp_y[0])

                    fig.add_trace(go.Scatter(
                        x=exp_x, y=exp_y,
                        mode="lines",
                        fill="toself",
                        fillcolor=hex_to_rgba(c_color, 0.12),
                        line=dict(color=c_color, width=1.5, dash="dot"),
                        name=f"{c_name} Region",
                        hoverinfo="skip",
                        showlegend=False
                    ))
                except Exception:
                    pass

    elif zone_style == "Continuous KNN Grid":
        margin_x = (x.max() - x.min()) * 0.08
        margin_y = (y.max() - y.min()) * 0.08
        grid_x = np.linspace(x.min() - margin_x, x.max() + margin_x, 100)
        grid_y = np.linspace(y.min() - margin_y, y.max() + margin_y, 100)
        xx, yy = np.meshgrid(grid_x, grid_y)

        grid_points = np.c_[xx.ravel(), yy.ravel()]
        cluster_centers = np.array([
            df[df["cluster_name"] == c_name][["pca_x", "pca_y"]].mean().values
            for c_name in sorted_clusters
        ])

        dists = np.linalg.norm(grid_points[:, np.newaxis] - cluster_centers, axis=2)
        nearest_idx = np.argmin(dists, axis=1).reshape(xx.shape)

        n_c = len(sorted_clusters)
        colorscale = []
        for idx, c_name in enumerate(sorted_clusters):
            c_hex = cluster_color_map[c_name]
            v_start = idx / n_c
            v_end = (idx + 1) / n_c
            colorscale.append([v_start, hex_to_rgba(c_hex, 0.15)])
            colorscale.append([v_end, hex_to_rgba(c_hex, 0.15)])

        fig.add_trace(go.Heatmap(
            x=grid_x, y=grid_y, z=nearest_idx,
            colorscale=colorscale,
            showscale=False,
            hoverinfo="skip"
        ))

    # --- 2. Scatter Points Per Cluster ---
    for c_name in sorted_clusters:
        c_df = df[df["cluster_name"] == c_name]
        c_color = cluster_color_map[c_name]

        hover_texts = []
        for _, row in c_df.iterrows():
            p_name = row["Player"]
            sq = row.get("Squad", "N/A")
            po = row.get("Pos", "N/A")
            mins = int(row.get("minutes", 0))

            txt = (
                f"<b>{p_name}</b> ({sq} • {po})<br>"
                f"Archetype: <b>{c_name}</b> | Minutes: <b>{mins:,}'</b><br>"
            )
            if "goals_per90" in row:
                txt += (
                    f"Goals/90: <b>{row.get('goals_per90', 0):.2f}</b> | Assists/90: <b>{row.get('assists_per90', 0):.2f}</b><br>"
                    f"xG/90: <b>{row.get('xg_per90', 0):.2f}</b> | Tackles/90: <b>{row.get('tackles_per90', 0):.2f}</b>"
                )
            else:
                txt += (
                    f"Save %: <b>{row.get('save_pct', 0):.1f}%</b> | CS %: <b>{row.get('clean_sheet_pct', 0):.1f}%</b><br>"
                    f"Saves/90: <b>{row.get('saves_per90', 0):.2f}</b>"
                )
            hover_texts.append(txt)

        fig.add_trace(go.Scatter(
            x=c_df["pca_x"],
            y=c_df["pca_y"],
            mode="markers",
            name=c_name,
            text=hover_texts,
            hoverinfo="text",
            marker=dict(
                size=9,
                color=c_color,
                line=dict(width=0.8, color="rgba(255,255,255,0.8)"),
                opacity=0.85
            )
        ))

    # --- 4. Highlighted Players ---
    if highlight_players:
        for hp in highlight_players:
            hp_mask = df["Player"].str.contains(hp, case=False, na=False)
            if hp_mask.any():
                hp_row = df[hp_mask].iloc[0]
                fig.add_trace(go.Scatter(
                    x=[hp_row["pca_x"]],
                    y=[hp_row["pca_y"]],
                    mode="markers+text",
                    name=f"⭐ {hp_row['Player']}",
                    text=[f"  <b>{hp_row['Player']} ({hp_row['Squad']})</b>"],
                    textposition="top right",
                    textfont=dict(color="#FFD700", size=13),
                    hoverinfo="text",
                    hovertext=f"<b>⭐ {hp_row['Player']}</b><br>Team: {hp_row['Squad']}<br>Cluster: {hp_row['cluster_name']}",
                    marker=dict(
                        size=18,
                        symbol="star",
                        color="#FF0055",
                        line=dict(width=2.5, color="#FFFFFF")
                    ),
                    showlegend=False
                ))

    # --- 5. Figure Layout & Styling ---
    fig.update_layout(
        title=dict(text="<b>Premier League Playstyle Cluster Map (2D PCA Projection)</b>", font=dict(size=16, color="white")),
        font=dict(family="Inter, Outfit, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Arial, sans-serif", color="white"),
        dragmode="pan",
        xaxis=dict(
            title="<b>PCA Axis 1 — Attacking Threat & Creative Progression →</b>",
            color="#cbd5e1",
            gridcolor="#1e293b",
            showgrid=True,
            zerolinecolor="#334155"
        ),
        yaxis=dict(
            title="<b>PCA Axis 2 — Directness & Defensive Engagement →</b>",
            color="#cbd5e1",
            gridcolor="#1e293b",
            showgrid=True,
            zerolinecolor="#334155"
        ),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0f172a",
        height=600,
        margin=dict(l=40, r=40, t=50, b=50),
        hoverlabel=dict(bgcolor="#1e293b", font_size=12, font_family="sans-serif"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(color="white", size=11),
            bgcolor="rgba(15, 23, 42, 0.85)",
            bordercolor="#334155",
            borderwidth=1
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
    # --- Inject Google European Fonts (Inter & Outfit) ---
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;700&display=swap');
            
            html, body, [class*="css"], .stMarkdown, .stSelectbox, .stMultiSelect, .stSlider, div {
                font-family: 'Inter', 'Outfit', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif !important;
            }
            
            h1, h2, h3, h4, h5, h6 {
                font-family: 'Outfit', 'Inter', sans-serif !important;
                letter-spacing: -0.02em;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

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

    player_type = st.sidebar.radio("Player Type", ["Outfield", "Goalkeeper"], key="player_type_radio")

    if player_type == "Outfield":
        df = outfield
        df_scaled = outfield_scaled
        feature_cols = [c for c in config.OUTFIELD_FEATURES if c in df.columns]
    else:
        df = gk
        df_scaled = gk_scaled
        feature_cols = [c for c in config.GK_FEATURES if c in df.columns]

    # Sorted list of all players (Aaron Hickey #1 at top, Anselmino moved to bottom)
    all_players = get_sorted_players(df)

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
    with st.sidebar.expander("📖 Quick Stat & Role Guide"):
        st.caption("Positions & Role Abbreviations:")
        st.markdown("**GK**: Goalkeeper (Shot-Stopper / Sweeper)")
        st.markdown("**DF / CB**: Center Back (Ball-Playing / Stopper)")
        st.markdown("**FB / WB**: Fullback / Wingback (Inverted / Overlapping)")
        st.markdown("**MF / DM**: Defensive Midfielder (Holding / Regista / Destroyer)")
        st.markdown("**MF / CM / B2B**: Central / Box-to-Box Midfielder")
        st.markdown("**AM / #10**: Attacking Midfielder / Playmaker")
        st.markdown("**FW / W / ST**: Winger / Striker (Inside Forward / False 9)")
        st.divider()
        st.caption("Key Metric Abbreviations:")
        st.markdown("**xG / 90**: Expected Goals per 90 mins")
        st.markdown("**xA / 90**: Expected Assists per 90 mins")
        st.markdown("**SCA / 90**: Shot-Creating Actions per 90 mins")
        st.markdown("**GCA / 90**: Goal-Creating Actions per 90 mins")
        st.markdown("**KP / 90**: Key Passes leading directly to shots")
        st.markdown("**PrgP / 90**: Progressive Passes >= 10 yards forward")
        st.markdown("**PrgC / 90**: Progressive Carries >= 10 yards forward")
        st.markdown("**PSxG +/-**: Post-shot xG minus goals allowed")

    # Sidebar About Expander
    with st.sidebar.expander("👨‍💻 About & Links"):
        st.markdown("**Created by:** Kotamraju Raman Karthik")
        st.markdown("🔗 [GitHub Repository](https://github.com/K-RamanKarthik/prem-league-project)")
        st.markdown("🔗 [LinkedIn Profile](https://www.linkedin.com/in/kotamraju-raman-karthik/)")

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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Radar Comparison", "👥 Similar Players", "🗺️ Cluster Map", "📖 Full Stats & Guide", "ℹ️ About"
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
        st.caption("Explore 2D PCA playstyle projection of Premier League players. Regions represent distinct tactical playstyle archetypes.")

        # Interactive Controls
        c_col1, c_col2 = st.columns([3, 2])
        with c_col1:
            map_highlights = st.multiselect(
                "⭐ Highlight Players on Map",
                all_players,
                default=[selected_player] if selected_player in all_players else [],
                key="cluster_map_highlights"
            )
        with c_col2:
            zone_style = st.selectbox(
                "🎨 Playstyle Region Display",
                ["Convex Hull Bubbles", "Continuous KNN Grid", "Points Only (No Regions)"],
                index=0,
                key="cluster_map_zone_style"
            )

        # Render Cluster Map
        fig_cluster = create_interactive_cluster_map(
            df,
            highlight_players=map_highlights,
            zone_style=zone_style
        )
        st.plotly_chart(
            fig_cluster,
            use_container_width=True,
            config={
                "scrollZoom": True,
                "displayModeBar": False
            }
        )

        st.divider()

        # Archetype Breakdown Cards
        st.subheader("📊 Tactical Archetypes Breakdown")
        clusters_in_df = sorted(df["cluster_name"].unique())
        card_cols = st.columns(min(len(clusters_in_df), 4))

        for idx, c_name in enumerate(clusters_in_df):
            col = card_cols[idx % len(card_cols)]
            c_sub = df[df["cluster_name"] == c_name]
            top_player = c_sub.sort_values("minutes", ascending=False).iloc[0]["Player"] if not c_sub.empty else "N/A"
            c_color = CLUSTER_COLORS[idx % len(CLUSTER_COLORS)]

            with col:
                st.markdown(f"""
                <div style="background-color: #1e293b; border-radius: 8px; padding: 12px; margin-bottom: 10px; border-left: 4px solid {c_color};">
                    <h4 style="margin: 0; color: white; font-size: 14px;">{c_name}</h4>
                    <p style="margin: 4px 0; font-size: 20px; font-weight: bold; color: {c_color};">{len(c_sub)} <span style="font-size: 11px; color: #94a3b8;">players</span></p>
                    <p style="margin: 0; font-size: 11px; color: #cbd5e1;"><b>Top Rep:</b> {top_player}</p>
                    <p style="margin: 0; font-size: 11px; color: #94a3b8;">Avg Mins: {int(c_sub['minutes'].mean()):,}'</p>
                </div>
                """, unsafe_allow_html=True)

        # Expandable Dataframe
        with st.expander("📋 View Complete Cluster Summary Table"):
            summary_df = (
                df.groupby("cluster_name")
                .agg(
                    Players=("Player", "count"),
                    Avg_Minutes=("minutes", "mean"),
                )
                .round(0)
                .sort_values("Players", ascending=False)
            )
            st.dataframe(summary_df, use_container_width=True)

    # ---- Tab 4: Full Stats & Complete Stat Guide ----
    with tab4:
        st.subheader(f"📋 Full Stats Profile — {selected_player}")

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

        # Section 1: Position Abbreviations & Tactical Roles Guide
        st.subheader("⚽ Premier League Position Abbreviations & Tactical Roles")
        st.caption("Comprehensive guide to position codes and modern tactical role archetypes.")

        role_col1, role_col2 = st.columns(2)

        with role_col1:
            st.markdown("""
            #### 🛡️ Defensive Positions & Roles
            - **GK (Goalkeeper)**:
              - **Shot-Stopper**: Reaction saves, high Save % and positive PSxG +/-.
              - **Sweeper-Keeper**: High defensive action distance and long distribution launches.
            - **DF / CB (Center Back)**:
              - **Ball-Playing Defender (BPD)**: High progressive passes (`PrgP/90`) and pass accuracy.
              - **Stopper / Anchor**: High tackles (`Tkl/90`), blocks (`Blk/90`), and aerial duels (`Aer Won/90`).
            - **FB / WB (Full Back / Wing Back)**:
              - **Inverted Fullback**: Tucks into central midfield during team build-up.
              - **Overlapping Wingback**: High progressive carries (`PrgC/90`), key passes (`KP/90`), and crosses.
            """)

        with role_col2:
            st.markdown("""
            #### ⚔️ Midfield & Forward Positions & Roles
            - **DM / MF (Defensive Midfielder)**:
              - **Holding / Regista**: Deep-lying playmaker controlling game tempo with high touch volume.
              - **Destroyer / Ball-Winner**: Aggressive pressing, tackles (`Tkl/90`), and interceptions (`Int/90`).
            - **CM / B2B (Central / Box-to-Box Midfielder)**:
              - **Engine / B2B**: High volume touches, box-to-box movement, and progressive actions.
            - **AM / CAM / #10 (Attacking Midfielder)**:
              - **Creative Playmaker**: High key passes (`KP/90`), xA/90, and shot-creating actions (`SCA/90`).
              - **Shadow Striker**: Operates near penalty box with high xG/90 and goal-creating actions (`GCA/90`).
            - **FW / W / ST (Winger / Striker)**:
              - **Inside Forward / Winger**: Dribbles past defenders (`Succ Drib/90`) and cuts inside to shoot.
              - **Target Man / Striker (False 9)**: Physical presence, aerial dominance, and goal finishing.
            """)

        st.divider()

        # Section 2: Playstyle Cluster Archetypes
        st.subheader("🏷️ Scouting Playstyle Cluster Archetypes")
        st.markdown("""
        The machine learning model categorizes Premier League players into 6 playstyle archetypes based on K-Means clustering:

        - 🔵 **Possession-Based + Playmaking**: Controllers who maintain high pass completion (`pass_completion_pct`) and dictate play.
        - 🟢 **Progressive Passing + Playmaking**: Midfield orchestrators who break defensive lines with long-range forward passes (`progressive_passes_per90`).
        - 🟡 **Aerial + Dribbling / Direct Threat**: Direct physical players excelling in 1v1 take-ons (`successful_dribbles_per90`) and aerial duels (`aerials_won_per90`).
        - 🔴 **Goal-Scoring + Goal-Creating**: Primary attackers with high expected goals (`xg_per90`), shot volume, and goal-creating actions (`gca_per90`).
        - 🟣 **Shot-Stopping Goalkeeper**: Goalkeepers evaluated on post-shot expected goals saved (`psxg_minus_ga`) and save percentage (`save_pct`).
        - 🟠 **Sweeper / Distributing Keeper**: Goalkeepers active outside the penalty box with long range launches and high defensive action distance.
        """)

        st.divider()

        # Section 3: Stat Glossary Reference Table
        st.subheader("📖 Complete Stat Metrics & Abbreviations Glossary")
        stat_table_rows = []
        for key, desc in STAT_GLOSSARY.items():
            abbrev = key.replace("_per90", "/90").replace("_pct", "%").replace("_", " ").title()
            cat = "Goalkeeping" if any(k in key for k in ["gk", "save", "psxg", "clean", "crosses", "goals_against"]) else (
                "Attacking" if any(k in key for k in ["goal", "xg", "shot", "gca"]) else (
                    "Playmaking" if any(k in key for k in ["pass", "xa", "sca", "carry", "dribble", "key"]) else "Defensive"
                )
            )
            stat_table_rows.append({"Abbreviation / Metric": abbrev, "Category": cat, "Definition & Guide": desc})

        stat_guide_df = pd.DataFrame(stat_table_rows)
        st.dataframe(stat_guide_df, use_container_width=True, hide_index=True)

    # ---- Tab 5: About & Links ----
    with tab5:
        st.subheader("ℹ️ About the Project & Creator")
        st.markdown("""
        **Premier League Player Scouting System (2025-26)** is an AI/ML powered football analytics application 
        that clusters players by playstyle archetypes and identifies statistical equivalents across the Premier League.
        
        #### 👨‍💻 Creator & Links:
        - **Author**: Kotamraju Raman Karthik
        - 🐙 **GitHub**: [github.com/K-RamanKarthik/prem-league-project](https://github.com/K-RamanKarthik/prem-league-project)
        - 💼 **LinkedIn**: [linkedin.com/in/kotamraju-raman-karthik/](https://www.linkedin.com/in/kotamraju-raman-karthik/)
        """)

    # ---- Footer ----
    st.divider()
    st.markdown(
        "Data: Premier League 2025-26 | Built by [Kotamraju Raman Karthik](https://www.linkedin.com/in/kotamraju-raman-karthik/) | "
        "[GitHub Repository](https://github.com/K-RamanKarthik/prem-league-project)"
    )


if __name__ == "__main__":
    main()
